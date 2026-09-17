/**
 * @file DeviceGraphOrchestratorGeneration.cpp
 * @brief Ordinary captured generation using the existing request/controller owner.
 *
 * Prefill sampling consumes no KV row. Each subsequent retained transaction
 * consumes the pending device token, then publishes one response and its next
 * frontier. No host token, speculative verifier or fake forward participates.
 * Sampler recordings borrow arena storage; their parent retires first. HIP
 * retains one complete transaction, never a host-submitted list of model stages.
 */
#include "DeviceGraphOrchestrator.h"
#include "backends/BackendManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "utils/Logger.h"

namespace llaminar2
{
bool DeviceGraphOrchestrator::observeOrdinaryGenerationForwardCompletion(int64_t completed_invocations)
{
    const auto &loop = mtp_device_generation_loop_graph_;
    std::string error;
    if (!forward_engine_ || !loop.ordinary_forward_identity || loop.source_fragments.size() != 1 ||
        !forward_engine_->observeComposedForwardCompletion(*loop.ordinary_forward_identity,
            loop.source_fragments.front().capture, completed_invocations, &error))
    {
        LOG_ERROR("Ordinary generation terminal lost its captured forward identity: " << error);
        return false;
    }
    return true;
}

std::optional<DeviceGraphOrchestrator::OrdinaryGenerationForward>
DeviceGraphOrchestrator::materializeOrdinaryGenerationForward(
    const int32_t *condition, std::string *error)
{
    const auto fail = [&](const std::string &reason) -> std::optional<OrdinaryGenerationForward> {
        if (error) *error = reason;
        return std::nullopt;
    };
    auto *const backend = getBackendFor(state_.device_id);
    void *const stream = mtp_device_generation_loop_graph_.stream.get();
    if (!backend || !stream || !condition || !state_.kv_cache || !request_position_ids_dev_)
        return fail("Resident ordinary forward requires an exact stream, token and owned KV position");
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
    input.graph_submission_intent = ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;
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
    const auto forward = forward_engine_->deviceLoopGraphTemplate(*signature, &detail);
    if (!forward)
        return fail("Ordinary generation requires a complete captured forward: " + detail);
    return OrdinaryGenerationForward{*forward, *signature};
}

bool DeviceGraphOrchestrator::materializeOrdinaryDeviceGenerationLoopGraph(
    DeviceGenerationSamplingMode sampling_mode, std::string *error,
    OrdinaryGenerationComposition composition)
{
    using namespace sampling_math;
    using SamplerStage = OrdinaryGenerationSamplingStage;
    using Plan = OrdinaryGenerationGraphPlan;
    auto &loop = mtp_device_generation_loop_graph_;
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
            (!pp_stage_config_->has_embedding && composition != OrdinaryGenerationComposition::PipelineTail))) ||
        moe_overlay_epoch_execution_binding_ || usesParticipantLocalDeviceMoERebalanceController())
        return fail("Ordinary generation requires an admitted scalar sampler and complete local graph; pipeline/overlay composition is not yet installed");

    const auto &admission = *active_device_generation_admission_;
    const auto &law = *admission.ordinary_sampling;
    const bool greedy = law.is_greedy();
    if ((greedy ? DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic) != sampling_mode)
        return fail("Ordinary materialization differs from the admitted sampling law");
    if (law.presence_penalty != mtp_request_penalty_policy_.presence_penalty ||
        law.frequency_penalty != mtp_request_penalty_policy_.frequency_penalty)
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

    const auto prepared = materializeOrdinaryGenerationForward(logical.next_condition_tokens_device, error);
    if (!prepared) return false;
    const auto &signature = prepared->signature;
    const auto &forward = prepared->graph;
    std::string detail;

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
        loop.source_fragments.size() == 1 && loop.source_fragments.front().capture == forward.capture)
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
    const DeviceControlledLoopFragment iteration[] = {
        {.name = "ordinary_resident_forward", .capture = forward.capture},
        {.name = "ordinary_decode_sample", .capture = loop.ordinary_children[Plan::decode_sampler].get()}};
    if (composition == OrdinaryGenerationComposition::PipelineTail)
    {
        // The rank installs activation/command edges around these recordings.
        // A local-only tail loop would advance KV without upstream layers.
    }
    else if (!hosted)
    {
        if (!DeviceGenerationGraphProgram::native(*loop.capture, control,
                {.initialization = initialization, .iteration = iteration}, detail))
            return fail(detail);
    }
    else
    {
        auto &transaction = loop.ordinary_children[Plan::hosted_transaction];
        transaction = context.createGraphCapture(stream);
        const GPUOrderedTimelineStep steps[] = {
            {.name = "ordinary_resident_forward", .capture = forward.capture},
            {.name = "ordinary_decode_sample", .capture = loop.ordinary_children[Plan::decode_sampler].get()}};
        if (!transaction || !transaction->buildOrderedTimelineTransaction(steps) || !transaction->instantiate() ||
            !loop.ordinary_children[Plan::prefill_sampler]->instantiate() ||
            !DeviceGenerationGraphProgram::ticketPublisher(*loop.capture, context, *backend, state_.device_id,
                control, nullptr, device_generation_storage_.dispatch_tickets_device, detail))
            return fail("Ordinary hosted graph compilation failed: " + detail);
        loop.hosted_fragments.push_back({.name = "ordinary_complete_transaction",
            .kind = HostedDeviceGenerationFragment::Kind::CapturedLocal, .capture = transaction.get()});
        loop.branch_fragment_counts[0] = 1;
    }
    // The forward identity and bindings are immutable; all mutable counters,
    // seeds and frontier contents stay in their existing arena-owned banks.
    loop.source_fragments.push_back(iteration[0]);
    loop.ordinary_sampling_identity = params;
    loop.ordinary_forward_identity = signature;
    loop.ordinary_leading = admission.initial_leading_row_disposition;
    loop.workspace_generation = generation;
    loop.request_count = 1;
    loop.depth_policy_mode = static_cast<int>(DeviceGenerationPolicyMode::Ordinary);
    loop.sampling_mode = sampling_mode;
    loop.fragment_count = hosted ? 1 : 3;
    loop.execution_kind = hosted ? MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedDispatchTicketPublisher
                                : MTPDeviceGenerationLoopGraphCache::ExecutionKind::NativeConditionalParent;
    loop.valid = true;
    PerfStatsCollector::addCounter("generation", "ordinary_graph_materializations", 1.0,
        "graph_setup", state_.device_id.toString());
    return true;
}
}
