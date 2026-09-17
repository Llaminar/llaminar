/**
 * @file PipelineDeviceGeneration.cpp
 * @brief Compose real local stage forwards around a single tail generation owner.
 *
 * Command fields borrow the existing request arena; followers never initialize
 * or inspect a response ledger. Their only overwritten words are health,
 * completion and the next input token, delivered by captured collectives.
 * Parent ownership and request-reset events stay with each DGO. The rank owns
 * communicator membership, setup fanout and one final response observation.
 */
#include "PipelineDeviceGeneration.h"
#include "PipelineForwardGraphEdges.h"
#include "DeviceGraphOrchestrator.h"
#include "TPWorkerPool.h"
#include "backends/BackendManager.h"
#include "collective/LocalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "execution/mtp/MTPServingForwardCaptureGeometry.h"
#include <barrier>
#include <algorithm>
#include <array>
#include <atomic>
#include <set>
#include <stdexcept>

namespace llaminar2
{
namespace
{
/** @brief Reject setup failure before any inference graph can be launched. */
bool collectPreparation(TPWorkerPool &workers)
{
    // The backend bounds each collective rendezvous. Preparing an entire
    // model can include many healthy collectives, so use the shared worker-
    // join policy rather than imposing a second whole-model deadline here.
    for (const auto &result : workers.collectAll(
            collective_timeout_policy::effectiveWorkerJoinTimeoutMs(0)))
    {
        if (!result.completed)
            throw std::runtime_error("Pipeline generation preparation did not join every participant");
        if (result.exception) std::rethrow_exception(result.exception);
        if (!result.success) return false;
    }
    return true;
}
}

PipelineDeviceGeneration::PipelineDeviceGeneration(
    std::span<const std::unique_ptr<IInferenceRunner>> stages)
{
    if (stages.size() < 2)
        throw std::invalid_argument("Pipeline generation requires at least two ordered stages");
    std::vector<GlobalDeviceAddress> devices;
    std::set<DeviceId> unique;
    for (size_t i = 0; i < stages.size(); ++i)
    {
        auto *stage = dynamic_cast<DeviceGraphOrchestrator *>(stages[i].get());
        if (!stage || !stage->pp_stage_config_ ||
            stage->pp_stage_config_->has_embedding != (i == 0) ||
            stage->pp_stage_config_->has_lm_head != (i + 1 == stages.size()) ||
            !stage->state_.hidden || !stage->state_.kv_cache ||
            !stage->mtp_device_generation_loop_graph_.capture ||
            stage->deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth) ==
                DeviceGenerationExecutionPolicy::Unsupported ||
            !unique.insert(stage->state_.device_id).second ||
            (!stages_.empty() && (stage->state_.device_id.type != stages_.front()->state_.device_id.type ||
                stage->graph_builder_->config().d_model != stages_.front()->graph_builder_->config().d_model ||
                stage->deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth) != execution_policy_)))
            throw std::invalid_argument("Pipeline generation requires distinct homogeneous captured stages with exact head/tail ownership; heterogeneous and nested TP composition is not installed");
        execution_policy_ = stage->deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth);
        stages_.push_back(stage);
        devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(stage->state_.device_id));
    }
    collective_ = std::make_unique<LocalTPContext>(std::move(devices), std::vector<float>{}, CollectiveBackendType::AUTO);
    if (!collective_) throw std::runtime_error("Pipeline generation collective admission failed");
    workers_ = std::make_unique<TPWorkerPool>(stages_.size());
    workers_->setFailureCallback([this] { collective_->requestAbort(); });
}

PipelineDeviceGeneration::~PipelineDeviceGeneration() = default;

bool PipelineDeviceGeneration::isHomogeneousGPUStageSet(std::span<const std::unique_ptr<IInferenceRunner>> stages)
{
    if (stages.size() < 2 || !stages.front() || !stages.front()->primaryDeviceId().is_gpu()) return false;
    const auto type = stages.front()->primaryDeviceId().type;
    return std::all_of(stages.begin(), stages.end(), [type](const auto &stage) {
        return stage && dynamic_cast<DeviceGraphOrchestrator *>(stage.get()) && stage->primaryDeviceId().type == type;
    });
}

bool PipelineDeviceGeneration::prepareServing(const ServingGraphFamilyMaterializationPlan &plan)
{
    if (phase_ != Phase::Idle || !plan.valid()) return false;
    phase_ = Phase::Failed;
    if (forward_edges_.empty())
    {
        // All stages share the rank's admitted ledger. The native boundary
        // owns one fence word per GPU, not a second activation/FP16 workspace.
        const auto &memory = stages_.front()->physical_memory_authority_;
        for (auto *stage : stages_)
            if (!memory || stage->physical_memory_authority_ != memory)
                throw std::logic_error("Pipeline capture requires one shared physical-memory authority");
        if (!collective_->reserveGraphCaptureBoundaryResources(memory)) return false;
        // A late edge install would change already captured topology. Reject
        // it before mutating any stage instead of invalidating/recapturing a
        // supposedly ready model. The owner is installed at serving setup.
        for (auto *stage : stages_)
            if (stage->forward_engine_ || stage->pipeline_forward_edges_)
                throw std::logic_error("Pipeline prefill transport must be frozen before forward-engine construction");
        for (size_t index = 0; index < stages_.size(); ++index)
        {
            auto &stage = *stages_[index];
            std::optional<PipelineForwardGraphEdges::FollowerState> follower;
            if (index + 1 < stages_.size() && stage.graph_builder_->config().mtp.enabled)
            {
                if (stage.mtp_publication_main_kv_base_checkpoints_.empty())
                    throw std::logic_error("Pipeline verifier lacks its admitted local KV checkpoint");
                const auto &checkpoint = stage.mtp_publication_main_kv_base_checkpoints_.front();
                if (!checkpoint.valid() || checkpoint.sequence_index != 0 || checkpoint.device != stage.state_.device_id)
                    throw std::logic_error("Pipeline verifier checkpoint has stale or foreign ownership");
                follower = PipelineForwardGraphEdges::FollowerState{
                    .backend = getBackendFor(stage.state_.device_id),
                    .checkpoint = {.cache = stage.state_.kv_cache.get(), .sequence_index = 0,
                        .checkpoint_device = checkpoint.data(), .checkpoint_bytes = checkpoint.bytes}};
            }
            forward_edges_.push_back(std::make_unique<PipelineForwardGraphEdges>(*collective_,
                static_cast<int>(index), *stage.state_.hidden, stage.graph_builder_->config().d_model, follower));
            stage.pipeline_forward_edges_ = forward_edges_.back().get();
        }
    }
    workers_->dispatch([&, this](size_t index) {
        auto local = plan;
        // The receive node fills the stage's own bank. No external tensor is
        // migrated or aliased into this participant's captured model graph.
        local.pipeline_hidden_input = index ? stages_[index]->state_.hidden.get() : nullptr;
        return stages_[index]->materializeServingGraphFamilyWithoutLaunch(local);
    });
    if (!collectPreparation(*workers_)) return false;
    workers_->dispatch([&, this](size_t index) {
        auto &stage = *stages_[index];
        if (!stage.graph_builder_->config().mtp.enabled) return true;
        auto metadata = stage.mtp_spec_decode_metadata_binding_.devicePointers();
        if (!stage.device_resident_logical_sequence_state_storage_.bindPublicationOutputs(&metadata, 1))
            return false;
        auto &gpu = GPUDeviceContextPool::instance().getContext(stage.state_.device_id);
        auto *context = stage.getDeviceContext(stage.state_.device_id);
        if (!context) return false;
        stage.pipeline_publication_transport_ = forward_edges_[index]->materializePublicationTransport(
            {metadata.accepted_state_slot_indices, metadata.target_cached_tokens,
                metadata.accepted_state_counts, metadata.publication_ok_flags}, stage.executor_, *context, gpu);
        return static_cast<bool>(stage.pipeline_publication_transport_);
    });
    if (!collectPreparation(*workers_)) return false;
    phase_ = Phase::Idle;
    return true;
}

bool PipelineDeviceGeneration::prefill(const int *tokens, int count,
    const PrefillChunkSchedulerPolicy &policy, int pad_token, bool allow_padding)
{
    if (phase_ != Phase::Idle || !prefillPrepared() || !tokens || count <= 0) return false;
    phase_ = Phase::Failed;
    workers_->dispatch([&, this](size_t index) {
        return stages_[index]->forwardPrefillChunkSchedule(tokens, count, policy, pad_token, allow_padding);
    });
    if (!collectPreparation(*workers_)) return false;
    phase_ = Phase::Idle;
    return true;
}

bool PipelineDeviceGeneration::forwardMTPCondition(MTPConditionForwardPurpose purpose,
    const std::function<bool(IInferenceRunner &)> &terminal)
{
    return forwardMTP(purpose, terminal);
}

bool PipelineDeviceGeneration::forwardMTPVerifier(int logical_rows,
    const std::function<bool(IInferenceRunner &)> &terminal)
{
    auto &tail = *stages_.back();
    std::string error;
    const auto width_policy = tail.mtpVerifierPhysicalWidthPolicyForRequest(1, &error);
    const auto geometry = resolveMTPServingForwardCaptureGeometry(tail.graph_builder_->config().mtp,
        tail.mtp_max_verifier_rows_);
    const int physical_rows = width_policy ? mtpVerifierPhysicalPaddedSeqLen(
        1, logical_rows, tail.mtp_max_verifier_rows_, *width_policy) : 0;
    if (!geometry.enabled || !geometry.valid() || logical_rows < 2 || physical_rows != geometry.verifier_rows)
    {
        LOG_ERROR("Pipeline verifier does not match its retained physical envelope: " << error);
        return false;
    }
    return forwardMTP(tail.mtp_verifier_outcome_graph_mode_, terminal);
}

bool PipelineDeviceGeneration::forwardMTP(const MTPMainForwardPolicy &policy,
    const std::function<bool(IInferenceRunner &)> &terminal)
{
    if ((phase_ != Phase::Idle && phase_ != Phase::Admitted) || !prefillPrepared() || !terminal)
        return false;
    const auto previous = phase_;
    phase_ = Phase::Failed;
    workers_->dispatch([&, this](size_t index) {
        auto &stage = *stages_[index];
        if (index + 1 == stages_.size()) return terminal(stage);
        std::string error;
        const bool ok = stage.executeMTPMainForward(policy,
            index ? stage.state_.hidden.get() : nullptr,
            stage.explicitGPUStreamForOperation("pipeline_mtp_main_forward"),
            ForwardGraphSubmissionIntent::Execute, &error);
        if (!ok) LOG_ERROR("Pipeline follower main forward failed: " << error);
        if (!ok)
            return false;

        /*
         * The terminal stage alone owns sampling.  A follower nevertheless
         * runs the same main-condition graph and therefore publishes a
         * replicated main-logits stream handoff.  No later pipeline stage may
         * consume that local replica: its activation edge, not its logits,
         * feeds the next participant.  Retire the handoff now so the next
         * captured condition graph can publish its own producer.  Retirement
         * merely drops an unused ordering token; it is not a stream
         * synchronization or a device-to-host observation.
         */
        // A PP follower's condition graph may be retained solely for its
        // activation send and native collective ordering; in that form it has
        // no public logits surface to consume.  Disarming its one-shot
        // deferral explicitly clears a possible completion handoff without
        // manufacturing a logits read.  The next captured collective still
        // elects its own asynchronous completion policy independently.
        stage.setMTPMainDecodeSyncDeferralEnabled(false);
        return true;
    });
    if (!collectPreparation(*workers_)) return false;
    phase_ = previous;
    return true;
}

bool PipelineDeviceGeneration::begin(const DeviceGenerationAdmissionRequest &request)
{
    if (phase_ != Phase::Idle || !request.valid() || request.request_count != 1 ||
        !request.depth_policy.isOrdinary())
        return false;
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        const auto &stage = *stages_[i];
        if (!stage.pp_stage_config_ ||
            stage.pp_stage_config_->has_embedding != (i == 0) ||
            stage.pp_stage_config_->has_lm_head != (i + 1 == stages_.size()) ||
            stage.state_.device_id != collective_->devices()[i].toLocalDeviceId())
            throw std::logic_error("Pipeline generation stage ownership changed after collective admission");
    }
    // Partial admission must never re-enter an unrelated rank/TP path. Until
    // every participant succeeds, this owner remains in an absorbing failure
    // state; the caller must retire the failed runner, not retry its request.
    phase_ = Phase::Failed;
    // The tail is the only owner permitted to initialize a sampler, response
    // budget or logical next-token authority. Followers admit local work only.
    if (!stages_.back()->beginDeviceResidentGeneration(request)) return false;
    for (size_t i = 0; i + 1 < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        void *stream = stage.explicitGPUStreamForOperation("pipeline_follower_admission");
        if (!stream || stage.device_generation_storage_.active_request_count != 0 ||
            stage.active_device_generation_admission_ || stage.mtp_device_generation_loop_graph_.launched ||
            !stage.device_generation_state_ready_.handoff.inactive() ||
            !stage.waitForPendingRequestStateReset(stream, DeviceTimelineRole::RequestAdmissionTransfer,
                "pipeline_follower_admission") ||
            !stage.waitForLiveInferenceStateReadyForObservation(stream,
                "pipeline_follower_admission", DeviceTimelineRole::DeviceGenerationController))
            return false;
        stage.device_generation_storage_.active_request_count = 1;
        // Do not reset the received command. Arrival unconditionally replaces
        // a previous terminal command before the first follower predicate.
        if (!stage.publishDeviceGenerationStateReady(stream, 1, DeviceGenerationStatePublicationKind::Admission))
            return false;
    }
    phase_ = Phase::Admitted;
    return true;
}

bool PipelineDeviceGeneration::prepareParticipants(DeviceGenerationSamplingMode sampling)
{
    workers_->dispatch([this, sampling](size_t index) {
        auto &stage = *stages_[index];
        std::string error;
        if (index + 1 == stages_.size())
        {
            const bool ready = stage.materializeOrdinaryDeviceGenerationLoopGraph(sampling, &error,
                DeviceGraphOrchestrator::OrdinaryGenerationComposition::PipelineTail);
            if (!ready) LOG_ERROR("Pipeline tail preparation failed: " << error);
            return ready;
        }
        const auto forward = stage.materializeOrdinaryGenerationForward(
            stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device, &error);
        if (!forward) { LOG_ERROR("Pipeline follower preparation failed: " << error); return false; }
        auto &loop = stage.mtp_device_generation_loop_graph_;
        const auto generation = stage.workspaceGeneration(stage.state_.device_id);
        if (loop.valid && loop.workspace_generation == generation &&
            loop.ordinary_forward_identity == forward->signature && loop.source_fragments.size() == 1 &&
            loop.source_fragments.front().capture == forward->graph.capture)
            return true;
        loop.invalidateGraph();
        loop.ordinary_forward_identity = forward->signature;
        loop.source_fragments.push_back({"pipeline_local_forward", forward->graph.capture});
        loop.workspace_generation = generation;
        loop.request_count = 1;
        using Kind = DeviceGraphOrchestrator::MTPDeviceGenerationLoopGraphCache::ExecutionKind;
        loop.execution_kind = execution_policy_ == DeviceGenerationExecutionPolicy::NativeConditionalGraph
            ? Kind::NativeConditionalParent : Kind::HostedPipelineTransaction;
        loop.valid = true;
        return true;
    });
    return collectPreparation(*workers_);
}

bool PipelineDeviceGeneration::composeParticipants()
{
    using namespace sampling_math;
    constexpr size_t command = 0, receive = 1, send = 2;
    const bool hosted = execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    std::barrier boundary(static_cast<ptrdiff_t>(stages_.size()));
    std::atomic<bool> failed{false};
    workers_->dispatch([&, this](size_t index) {
        auto &stage = *stages_[index];
        auto &loop = stage.mtp_device_generation_loop_graph_;
        auto &context = GPUDeviceContextPool::instance().getContext(stage.state_.device_id);
        void *const stream = loop.stream.get();
        const bool tail = index + 1 == stages_.size();
        loop.capture->reset();
        if (hosted && tail)
        {
            // Retire every enclosing executable before replacing its borrowed
            // recordings. A changed upstream identity recompiles the cohort.
            loop.hosted_fragments.clear();
            loop.ordinary_children[OrdinaryGenerationGraphPlan::hosted_transaction].reset();
        }
        for (auto &child : loop.pipeline_children) child.reset();
        // Exactly three words cross the command boundary; no response tokens,
        // verifier statistics, cache payload or host-maintained clock travels.
        const std::array fields{
            stage.device_generation_storage_.control_device + kDeviceGenerationControlOk,
            stage.device_generation_storage_.control_device + kDeviceGenerationControlRequestComplete,
            stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device};
        std::vector<LocalTPCollectiveSidebandBuffer> sidebands;
        for (auto *field : fields)
            sidebands.push_back({.kind = LocalTPCollectiveSidebandKind::Broadcast,
                .send_buffer = field, .recv_buffer = field, .element_count = 1,
                .dtype = CollectiveDataType::INT32, .root_device_index = static_cast<int>(stages_.size() - 1),
                .name = "pipeline_command"});
        const auto exchange = [&] { return collective_->collectiveSidebandSpanOnStream(
            sidebands, static_cast<int>(index), stream, "pipeline_generation_command"); };
        // Preparation phases stay symmetric even if a local capture fails.
        // This barrier is setup-only, outside every executable and token loop.
        const auto record = [&](std::unique_ptr<IGPUGraphCapture> &child, auto &&enqueue) {
            std::optional<ScopedBackendGraphCapture> capture;
            bool began = false;
            // An exception must not strand peers at the setup boundary. Finish
            // every cohort phase and close each successfully opened capture;
            // then return failure before any parent is submitted.
            try {
                child = context.createGraphCapture(stream);
                if (child) capture.emplace(context, *child, "pipeline_generation_collective");
                began = capture && capture->begin();
            } catch (const std::exception &error) {
                LOG_ERROR("Pipeline collective capture admission failed: " << error.what());
            }
            if (!began) failed = true;
            boundary.arrive_and_wait();
            const bool admitted = !failed.load();
            boundary.arrive_and_wait();
            try {
                if (admitted && !enqueue()) failed = true;
            } catch (const std::exception &error) {
                failed = true;
                collective_->requestAbort();
                LOG_ERROR("Pipeline collective recording failed: " << error.what());
            }
            boundary.arrive_and_wait();
            try {
                if (began) capture->finish();
            } catch (const std::exception &error) {
                failed = true;
                LOG_ERROR("Pipeline collective capture retirement failed: " << error.what());
            }
            boundary.arrive_and_wait();
        };
        record(loop.pipeline_children[command], exchange);
        for (size_t edge = 0; edge + 1 < stages_.size(); ++edge)
        {
            const bool sender = index == edge, receiver = index == edge + 1;
            std::unique_ptr<IGPUGraphCapture> empty;
            auto &child = sender ? loop.pipeline_children[send] : receiver ? loop.pipeline_children[receive] : empty;
            record(child, [&] {
                if (!sender && !receiver) return true;
                const CollectiveP2POp operation{
                    .kind = sender ? CollectiveP2POpKind::Send : CollectiveP2POpKind::Recv,
                    .send_buffer = sender ? stage.state_.hidden->gpu_data_ptr() : nullptr,
                    .recv_buffer = receiver ? stage.state_.hidden->gpu_data_ptr() : nullptr,
                    .count = static_cast<size_t>(stage.graph_builder_->config().d_model),
                    .dtype = CollectiveDataType::FLOAT32,
                    .peer = static_cast<int>(sender ? edge + 1 : edge)};
                return collective_->groupedP2PRawOnStream({operation}, static_cast<int>(index), stream,
                    "pipeline_generation_activation");
            });
        }
        if (failed.load()) return false;
        std::vector<DeviceControlledLoopFragment> body;
        // HIP submits a complete transaction only after a tail ticket. Command
        // arrival belongs inside that transaction, before any local KV work.
        // No initial-command executable or terminal stop broadcast is needed.
        if (hosted) body.push_back({"pipeline command arrival", loop.pipeline_children[command].get()});
        if (index > 0) body.push_back({"receive pipeline activation", loop.pipeline_children[receive].get()});
        body.push_back(loop.source_fragments.front());
        if (!tail) body.push_back({"send pipeline activation", loop.pipeline_children[send].get()});
        else body.push_back({"tail sampler", loop.ordinary_children[OrdinaryGenerationGraphPlan::decode_sampler].get()});
        if (hosted)
        {
            std::vector<GPUOrderedTimelineStep> steps;
            for (const auto &fragment : body)
                steps.push_back({.name = fragment.name, .capture = fragment.capture});
            auto &transaction = tail ? loop.ordinary_children[OrdinaryGenerationGraphPlan::hosted_transaction] : loop.capture;
            if (tail) transaction = context.createGraphCapture(stream);
            if (!transaction || !transaction->buildOrderedTimelineTransaction(steps) || !transaction->instantiate())
                return false;
            if (tail)
            {
                std::string detail;
                const DeviceGenerationGraphControl control{stage.active_device_generation_admission_->depth_policy,
                    stage.device_generation_storage_.control_device, stage.device_generation_storage_.control_stride, 1};
                if (!loop.ordinary_children[OrdinaryGenerationGraphPlan::prefill_sampler]->instantiate() ||
                    !DeviceGenerationGraphProgram::ticketPublisher(*loop.capture, context,
                        *getBackendFor(stage.state_.device_id), stage.state_.device_id, control, nullptr,
                        stage.device_generation_storage_.dispatch_tickets_device, detail))
                {
                    LOG_ERROR("Pipeline ticket publisher compilation failed: " << detail);
                    return false;
                }
                loop.hosted_fragments.push_back({.name = "pipeline_complete_transaction",
                    .kind = DeviceGraphOrchestrator::HostedDeviceGenerationFragment::Kind::CapturedLocal,
                    .capture = transaction.get()});
                loop.branch_fragment_counts[0] = 1;
            }
        }
        else
        {
            body.push_back({"next or terminal pipeline command", loop.pipeline_children[command].get()});
            const DeviceControlledLoopEntryFragment entry[] = {{"pipeline command arrival", loop.pipeline_children[command].get()}};
            const DeviceControlledLoopFragment initial[] = {{
                .name = "tail prefill sampler", .capture = loop.ordinary_children[OrdinaryGenerationGraphPlan::prefill_sampler].get(),
                .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordZero,
                .condition_word_device = reinterpret_cast<const uint32_t *>(
                    stage.device_generation_storage_.control_device + kDeviceGenerationControlNextLeadingCommittedOutputCount)}};
            const DeviceControlledLoopProgram program{.entry = entry,
                .initialization = tail ? std::span<const DeviceControlledLoopFragment>(initial) : std::span<const DeviceControlledLoopFragment>{},
                .iteration = body};
            // A follower's predicate is a received command, NOT an independently
            // admitted generation controller. Only the tail has sampler fragments.
            const DeviceControlledLoopPredicate predicate{
                .control_rows_device = stage.device_generation_storage_.control_device,
                .control_stride = stage.device_generation_storage_.control_stride, .request_count = 1,
                .healthy_index = kDeviceGenerationControlOk, .complete_index = kDeviceGenerationControlRequestComplete};
            if (!loop.capture->buildDeviceControlledWhileLoop(program, predicate) || !loop.capture->instantiate())
                return false;
        }
        loop.fragment_count = hosted ? body.size() : body.size() + 1 + (tail ? 1 : 0);
        PerfStatsCollector::addCounter("generation", "pipeline_graph_materializations", 1.0,
            "graph_setup", stage.state_.device_id.toString());
        return true;
    });
    return collectPreparation(*workers_);
}

bool PipelineDeviceGeneration::materialize(int requests, int depth,
    DeviceGenerationLoopTopology topology, DeviceGenerationSamplingMode sampling)
{
    if (phase_ != Phase::Admitted || requests != 1 || depth != 0 ||
        topology != DeviceGenerationLoopTopology::FixedDepth || !isValidDeviceGenerationSamplingMode(sampling))
        return false;
    phase_ = Phase::Failed;
    if (!prepareParticipants(sampling)) return false;
    const bool reusable = std::all_of(stages_.begin(), stages_.end(), [](auto *stage) {
        return stage->mtp_device_generation_loop_graph_.capture->hasExecutable(); });
    if (!reusable && !composeParticipants()) return false;
    phase_ = Phase::Materialized;
    return true;
}

bool PipelineDeviceGeneration::launch()
{
    if (phase_ != Phase::Materialized) return false;
    phase_ = Phase::Failed;
    if (execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)
    {
        if (!launchHostedTransactions())
        {
            collective_->requestAbort();
            throw std::runtime_error("Pipeline generation failed during ticket-selected distributed submission");
        }
        phase_ = Phase::Submitted;
        return true;
    }
    for (auto *stage : stages_)
        if (!stage->launchDeviceResidentGeneration())
        {
            collective_->requestAbort();
            throw std::runtime_error("Pipeline generation failed during distributed submission");
        }
    phase_ = Phase::Submitted;
    return true;
}

bool PipelineDeviceGeneration::launchHostedTransactions()
{
    auto &tail = *stages_.back();
    auto &tail_loop = tail.mtp_device_generation_loop_graph_;
    for (auto *stage : stages_)
    {
        auto &loop = stage->mtp_device_generation_loop_graph_;
        if (!loop.valid || loop.launched || !loop.capture->hasExecutable() ||
            loop.workspace_generation != stage->workspaceGeneration(stage->state_.device_id) ||
            !stage->consumeDeviceGenerationStateReady(loop.stream.get(),
                DeviceTimelineRole::DeviceGenerationController, 1, "pipeline_hosted_admission"))
            return false;
        loop.launched = true;
    }
    if (!tail.waitForDeviceResidentLogicalSequenceStateRowReuse(tail_loop.stream.get(),
            "pipeline_hosted_generation_writer") || !tail.hosted_device_generation_cursor_.startScheduler())
        return false;
    const auto submit_followers = [&] {
        for (size_t i = 0; i + 1 < stages_.size(); ++i)
        {
            auto &loop = stages_[i]->mtp_device_generation_loop_graph_;
            if (!loop.capture->launchOnStream(loop.stream.get())) return false;
        }
        return true;
    };
    // A fresh prefill owns one pending sample and needs no follower forward.
    // A continued response already emitted that sample, so every stage must
    // consume its retained frontier once before the first ticket is published.
    const bool pending = tail_loop.ordinary_leading == sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse;
    const auto initial = pending ? OrdinaryGenerationGraphPlan::prefill_sampler : OrdinaryGenerationGraphPlan::hosted_transaction;
    if ((!pending && !submit_followers()) ||
        !tail_loop.ordinary_children[initial]->launchOnStream(tail_loop.stream.get()) ||
        !tail.enqueueDeviceGenerationDispatchTicketObservation())
        return false;
    const int bound = tail.active_device_generation_admission_->max_new_tokens;
    for (int observation = 0; observation < bound; ++observation)
    {
        sampling_math::DeviceGenerationDispatchTicket ticket;
        if (!tail.observeDeviceGenerationDispatchTicket(&ticket) ||
            (!ticket.complete && !submit_followers()) ||
            !tail.submitHostScheduledDeviceGenerationAdvance(ticket))
            return false;
        if (ticket.complete)
        {
            // Tail authentication closes the distributed request. Followers
            // publish only their own stream completion, not invented terminal
            // controller words. This also retires a one-token, zero-forward
            // request without reading a stale follower command.
            for (size_t i = 0; i + 1 < stages_.size(); ++i)
                if (!stages_[i]->publishDeviceGenerationStateReady(
                        stages_[i]->mtp_device_generation_loop_graph_.stream.get(), 1,
                        DeviceGenerationStatePublicationKind::Terminal)) return false;
            return true;
        }
    }
    throw std::runtime_error("Pipeline generation exhausted its admitted response bound without a terminal ticket");
}

bool PipelineDeviceGeneration::finish(DeviceGenerationTerminalResult *result)
{
    if (phase_ != Phase::Submitted || !result) return false;
    phase_ = Phase::Failed;
    if (!stages_.back()->finishDeviceResidentGeneration(result)) return false;
    std::vector<std::array<int, 3>> terminal(stages_.size());
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto *backend = getBackendFor(stage.state_.device_id);
        void *stream = stage.mtp_device_generation_loop_graph_.stream.get();
        const bool follower = i + 1 < stages_.size();
        if (follower && !stage.consumeDeviceGenerationStateReady(stream, DeviceTimelineRole::HostResultBridge,
                1, "pipeline_follower_terminal")) return false;
        // One small terminal observation proves actual independent KV work.
        // It is never used to repair a stage or choose another transaction.
        const std::array<const int *, 3> sources{
            stage.state_.kv_cache->deviceSequenceCachedTokenCountPtr(0),
            stage.device_generation_storage_.control_device + sampling_math::kDeviceGenerationControlOk,
            stage.device_generation_storage_.control_device + sampling_math::kDeviceGenerationControlRequestComplete};
        const size_t observed_fields = follower &&
            execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions ? 1 : sources.size();
        for (size_t field = 0; field < observed_fields; ++field)
            if (!backend->deviceToHostOnStream(&terminal[i][field], sources[field], sizeof(int),
                    stage.state_.device_id.gpu_ordinal(), stream)) return false;
        if (!backend->recordEvent(stage.device_generation_terminal_host_ready_event_.get(),
                stage.state_.device_id.gpu_ordinal(), stream)) return false;
    }
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        const bool hosted_follower = i + 1 < stages_.size() &&
            execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
        if (!getBackendFor(stage.state_.device_id)->waitForEvent(stage.device_generation_terminal_host_ready_event_.get(),
                stage.state_.device_id.gpu_ordinal()) || (!hosted_follower && (terminal[i][1] != 1 || terminal[i][2] != 1)))
            return false;
        if (i + 1 < stages_.size())
        {
            if (!stage.mtp_device_generation_loop_graph_.retireCompletedLaunch() ||
                !stage.device_generation_state_ready_.handoff.retireAfterHostCompletion(1)) return false;
            stage.device_generation_storage_.active_request_count = 0;
        }
    }
    for (const auto &local : terminal)
        if (local[0] != terminal.back()[0])
            throw std::runtime_error("Pipeline generation terminal KV positions disagree; no state repair is permitted");
    // The tail reported its own completed forwards through the ordinary DGO
    // terminal. Followers report only after the independent KV check above;
    // they borrow its authenticated count, never a mirrored response ledger.
    for (size_t i = 0; i + 1 < stages_.size(); ++i)
        if (!stages_[i]->observeOrdinaryGenerationForwardCompletion(
                result->requests.front().published_state_commit_count)) return false;
    PerfStatsCollector::addCounter("generation", "pipeline_terminal_responses", 1.0, "decode", "rank",
        {{"response_owner", "tail"}, {"validation", "independent_stage_kv_positions"}});
    phase_ = Phase::Idle;
    return true;
}
}
