/**
 * @file PipelineDeviceGeneration.cpp
 * @brief Compose real local stage forwards around a single tail generation owner.
 *
 * Command fields borrow the existing request arena; followers never initialize
 * or inspect a response ledger. Their only overwritten words are health,
 * completion and the next input token, delivered by captured collectives.
 * Participant model graphs and request-reset events stay with each DGO. The
 * rank owns communicator membership, setup fanout, composed ForwardOnly parents
 * and one final response observation. Those parents retire while the DGO events
 * are live; transport and edge owners remain until the child graphs retire.
 * Captured-domain ForwardOnly parents borrow the complete existing scalar
 * metadata/activation/model body. Only terminal-domain participants publish a
 * full frontier; upstream domains close their actual partial row writer. No
 * sampler, host token replay or second transport/storage convention is added.
 */
#include "execution/qwen4exp/QwenSparseFollowerCounters.h"
#include "PipelineDeviceGeneration.h"
#include "PipelineForwardGraphEdges.h"
#include "DeviceGraphOrchestrator.h"
#include "IRankOrchestrator.h"
#include "RankOrchestrator.h"
#include "TPWorkerPool.h"
#include "backends/BackendManager.h"
#include "collective/LocalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "execution/mtp/MTPServingForwardCaptureGeometry.h"
#include "execution/mtp/MTPSpecDecodeMetadata.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "planning/PipelineTransferMemory.h"
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
        auto *rank = dynamic_cast<IRankOrchestrator *>(stages[i].get());
        const int count = rank ? rank->device_count() : 1;
        auto *context = rank ? rank->localTPContext() : nullptr;
        if (!stages[i] || count < 1 || (count > 1 && (!context || context->degree() != count)))
            throw std::invalid_argument("Pipeline generation requires exact native TP domain membership");
        domains_.push_back({stages[i].get(), context, stages_.size(), size_t(count)});
        for (int member = 0; member < count; ++member)
        {
            auto *stage = dynamic_cast<DeviceGraphOrchestrator *>(rank ? rank->deviceRunner(member) : stages[i].get());
            if (!stage || !stage->pp_stage_config_ ||
                stage->pp_stage_config_->has_embedding != (i == 0) ||
                stage->pp_stage_config_->has_lm_head != (i + 1 == stages.size()) ||
                !stage->state_.hidden || !stage->state_.kv_cache || !stage->arena_ ||
                !stage->deviceGenerationLoopGraph().capture ||
                stage->deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth) ==
                    DeviceGenerationExecutionPolicy::Unsupported ||
                !unique.insert(stage->state_.device_id).second ||
                (!stages_.empty() && stage->graph_builder_->config().residualWidth() != stages_.front()->graph_builder_->config().residualWidth()))
                throw std::invalid_argument("Pipeline generation requires distinct captured GPU members with exact layer-domain ownership");
            (void)PipelineNativeDomain(stage->state_.device_id, context, member);
            stages_.push_back(stage);
            devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(stage->state_.device_id));
        }
    }
    const bool one_native_pipeline = std::all_of(domains_.begin(), domains_.end(), [](const auto &domain) {
        return domain.count == 1;
    }) && std::all_of(stages_.begin(), stages_.end(), [&](const auto *stage) {
        return stage->state_.device_id.type == stages_.front()->state_.device_id.type;
    });
    if (one_native_pipeline)
    {
        collective_ = std::make_unique<LocalTPContext>(std::move(devices), std::vector<float>{}, CollectiveBackendType::AUTO);
        execution_policy_ = stages_.front()->deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth);
    }
    else
    {
        // A custom mapped channel is justified only at a genuine inter-vendor
        // boundary. Homogeneous domain-to-domain P2P needs its own native
        // composition; never silently substitute a host bounce for NCCL/RCCL.
        for (size_t i = 1; i < domains_.size(); ++i)
            if (stages_[domains_[i - 1].first]->state_.device_id.type == stages_[domains_[i].first]->state_.device_id.type)
                throw std::invalid_argument("Nested homogeneous pipeline domains require native domain-to-domain composition, not a heterogeneous channel");
        transport_ = Transport::CapturedDomains;
        execution_policy_ = DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    }
    workers_ = std::make_unique<TPWorkerPool>(stages_.size());
    workers_->setFailureCallback([this] { abortParticipants(); });
}

/** @brief Immutable complete parents borrowing original per-device model owners.
 * The boolean records host submission lifetime only, never device policy state.
 * Parents retire before their borrowed transport/publication child recordings. */
struct PipelineDeviceGeneration::ForwardOnlyFamily
{
    /** @brief Exact setup binding for one original native or captured-domain participant. */
    struct Participant
    {
        DeviceId device = DeviceId::invalid();
        uint64_t workspace = 0;
        void *stream = nullptr;
        const void *arena = nullptr;
        const IInferenceRunner *domain_runner = nullptr;
        const ILocalTPContext *domain_context = nullptr;
        const PipelineForwardGraphEdges *edges = nullptr;
        size_t domain_participant = 0;
        const void *hidden = nullptr;
        int width = 0;
        const void *logits = nullptr;
        const void *storage_boundary = nullptr;
        const IGPUGraphCapture *storage_terminal = nullptr;
        std::optional<DeviceGraphOrchestrator::OrdinaryGenerationForward> forward;
        sampling_math::OrdinaryGenerationPublication publication;
        std::array<std::unique_ptr<IGPUGraphCapture>, 5> children;
        std::unique_ptr<IGPUGraphCapture> parent;
        bool submitted = false;
        bool terminal_copies_pending = false; ///< Host-backing lifetime only, never GPU policy.
    };
    std::vector<Participant> participants;

    /** @brief Project exact arena-owned publisher bindings without reading device values. */
    static sampling_math::OrdinaryGenerationPublication publicationFor(const DeviceGraphOrchestrator &stage)
    {
        using namespace sampling_math;
        const auto &logical = stage.device_resident_logical_sequence_state_storage_;
        return {.request_count = 1,
            .sampled_tokens = static_cast<int32_t *>(stage.stochastic_target_sample_tokens_dev_),
            .stop_tokens = {static_cast<const int32_t *>(stage.mtp_verifier_stop_tokens_dev_), kSpeculativeBatchMaxStopTokens, 0},
            .source = OrdinaryGenerationSampleSource::DecodeLogits,
            .response_tokens = stage.device_generation_storage_.response_tokens_device,
            .response_token_stride = stage.device_generation_storage_.response_token_stride,
            .control = stage.device_generation_storage_.control_device,
            .control_stride = stage.device_generation_storage_.control_stride,
            .frontier = {logical.target_cached_tokens_device, logical.next_condition_tokens_device,
                logical.stopped_flags_device, logical.publication_ok_flags_device, 1, stage.state_.max_seq_len},
            .history = {static_cast<int32_t *>(stage.mtp_generated_token_counts_dev_), stage.state_.vocab_size,
                stage.state_.vocab_size, 1}};
    }
};

std::unique_ptr<PipelineDeviceGeneration::ForwardOnlyFamily> &PipelineDeviceGeneration::forwardOnlyFamily()
{
    return forward_only_families_[sparseQueryExecutionProgramIndex(
        stages_.back()->sparseQueryForwardProgram(ForwardExecutionPhase::Decode))];
}
const std::unique_ptr<PipelineDeviceGeneration::ForwardOnlyFamily> &PipelineDeviceGeneration::forwardOnlyFamily() const
{
    return forward_only_families_[sparseQueryExecutionProgramIndex(
        stages_.back()->sparseQueryForwardProgram(ForwardExecutionPhase::Decode))];
}

bool PipelineDeviceGeneration::forwardOnlyFamilyCurrent() const
{
    if (!forwardOnlyFamily() || forwardOnlyFamily()->participants.size() != stages_.size() ||
        (transport_ == Transport::NativePipeline && (!collective_ || domains_.size() != stages_.size()))) return false;
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        const auto &stage = *stages_[i];
        const auto &part = forwardOnlyFamily()->participants[i];
        std::string error;
        const auto &domain = domainFor(i);
        if ((transport_ == Transport::NativePipeline && domain.count != 1) ||
            (transport_ == Transport::CapturedDomains &&
             (part.domain_runner != domain.runner || part.domain_context != domain.context ||
              part.domain_participant != i - domain.first || part.edges != stage.pipeline_forward_edges_ ||
              !part.edges || !part.edges->hasHeterogeneousBoundary())) || part.device != stage.state_.device_id ||
            part.workspace != stage.workspaceGeneration(stage.state_.device_id) ||
            part.stream != stage.deviceGenerationLoopGraph().stream.get() ||
            part.arena != stage.arena_.get() ||
            part.hidden != stage.state_.hidden->gpu_data_ptr() ||
            part.width != stage.graph_builder_->config().residualWidth() || part.width <= 0 ||
            stage.state_.hidden->native_type() != TensorType::FP32 ||
            static_cast<size_t>(part.width) > stage.state_.hidden->numel() ||
            part.logits != stage.state_.logits->gpu_data_ptr() ||
            part.storage_boundary != stage.retained_storage_boundary_.get() ||
            part.storage_boundary != stage.graph_builder_->retainedStorageBoundary(stage.state_.device_id).get() ||
            part.storage_terminal != stage.retained_storage_terminal_capture_.get() ||
            part.publication != ForwardOnlyFamily::publicationFor(stage) ||
            !part.parent || !part.parent->hasExecutable() || part.parent->executionStream() != part.stream || !part.forward || !part.forward->native() ||
            part.forward->signature.sparse_query_execution_program != stage.sparseQueryForwardProgram(ForwardExecutionPhase::Decode) ||
            !stage.forward_engine_ || !stage.forward_engine_->observeComposedForwardCompletion(
                part.forward->signature, part.forward->native()->capture, 0, &error)) return false;
    }
    return true;
}

bool PipelineDeviceGeneration::prepareForwardOnlyFamily()
{
    // Captured domains retain their original metadata/activation enclosure;
    // native singleton PP retains its explicit communicator children below.
    if (transport_ == Transport::CapturedDomains)
        return prepareCapturedDomainForwardOnlyFamily();
    const auto count = stages_.front()->sparseQueryProgramCount();
    if (std::any_of(stages_.begin(), stages_.end(), [count](auto *stage) {
            return stage->sparseQueryProgramCount() != count; }))
        throw std::logic_error("Pipeline forced-query family disagrees with original model geometry");
    const auto compile_program = [&]() -> bool
    {
        if (forwardOnlyFamily()) return forwardOnlyFamilyCurrent();
        auto family = std::make_unique<ForwardOnlyFamily>();
        family->participants.resize(stages_.size());
        std::barrier boundary(static_cast<ptrdiff_t>(stages_.size()));
        std::atomic<bool> failed{false};
        workers_->dispatch([&, this](size_t index) {
            auto &stage = *stages_[index];
            auto &part = family->participants[index];
            IWorkerGPUContext *context = nullptr;
            // Every potentially rejecting setup operation is inside this boundary:
            // one failed member must still arrive at every cohort capture barrier.
            try {
                context = &GPUDeviceContextPool::instance().getContext(stage.state_.device_id);
                part.device = stage.state_.device_id;
                part.workspace = stage.workspaceGeneration(part.device);
                part.stream = stage.deviceGenerationLoopGraph().stream.get();
                part.arena = stage.arena_.get();
                part.hidden = stage.state_.hidden->gpu_data_ptr();
                part.width = stage.graph_builder_->config().residualWidth();
                part.logits = stage.state_.logits->gpu_data_ptr();
                std::string error;
                // Serving setup already retained this exact original resident-input body.
                // Export does not launch it or admit a controller/sampler/request.
                part.forward = stage.materializeOrdinaryGenerationForward(
                    stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device,
                    OrdinaryForwardRetentionKind::NativeComposable, &error);
                part.publication = ForwardOnlyFamily::publicationFor(stage);
                stage.prepareRetainedStorageBinding();
                (void)stage.retainedStorageTerminalFragments();
                part.storage_boundary = stage.retained_storage_boundary_.get();
                part.storage_terminal = stage.retained_storage_terminal_capture_.get();
                if (!part.workspace || !part.stream || !part.forward || !part.forward->native() ||
                    part.width <= 0 || stage.state_.hidden->native_type() != TensorType::FP32 ||
                    static_cast<size_t>(part.width) > stage.state_.hidden->numel() ||
                    !part.publication.valid()) failed = true;
            } catch (const std::exception &failure) {
                failed = true;
                LOG_ERROR("Pipeline ForwardOnly source preparation failed: " << failure.what());
            } catch (...) {
                failed = true;
                LOG_ERROR("Pipeline ForwardOnly source preparation failed with an unknown exception");
            }
            boundary.arrive_and_wait();
            // Every participant enters every capture phase, even on setup failure.
            // This is setup fanout only; no host token/row/control decides a replay.
            const auto record = [&](std::unique_ptr<IGPUGraphCapture> &child, auto &&enqueue) {
                std::optional<ScopedBackendGraphCapture> capture;
                bool began = false;
                try {
                    if (!failed.load() && context)
                    {
                        child = context->createGraphCapture(part.stream);
                        if (child) capture.emplace(*context, *child, "pipeline_forward_only_collective");
                        began = capture && capture->begin();
                    }
                } catch (const std::exception &failure) {
                    failed = true;
                    LOG_ERROR("Pipeline ForwardOnly capture admission failed: " << failure.what());
                } catch (...) { failed = true; }
                if (!began) failed = true;
                boundary.arrive_and_wait();
                const bool admitted = !failed.load();
                boundary.arrive_and_wait();
                try { if (admitted && !enqueue()) failed = true; }
                catch (const std::exception &failure) {
                    failed = true;
                    collective_->requestAbort();
                    LOG_ERROR("Pipeline ForwardOnly recording failed: " << failure.what());
                } catch (...) { failed = true; collective_->requestAbort(); }
                boundary.arrive_and_wait();
                try { if (began) capture->finish(); }
                catch (const std::exception &failure) {
                    failed = true;
                    LOG_ERROR("Pipeline ForwardOnly recording retirement failed: " << failure.what());
                } catch (...) { failed = true; }
                boundary.arrive_and_wait();
            };
            using namespace sampling_math;
            const auto exchange = [&](bool terminal) {
                std::vector<LocalTPCollectiveSidebandBuffer> fields;
                const auto add = [&](int32_t *field) {
                    fields.push_back({.kind = LocalTPCollectiveSidebandKind::Broadcast,
                        .send_buffer = field, .recv_buffer = field, .element_count = 1,
                        .dtype = CollectiveDataType::INT32, .root_device_index = static_cast<int>(stages_.size() - 1),
                        .name = "pipeline_forward_only_command"});
                };
                add(stage.device_generation_storage_.control_device + kDeviceGenerationControlRequestComplete);
                if (!terminal) add(stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device);
                // Terminal must not overwrite a follower's own failed health word.
                return collective_->collectiveSidebandSpanOnStream(fields, static_cast<int>(index),
                    part.stream, terminal ? "pipeline_forward_only_complete" : "pipeline_forward_only_input");
            };
            record(part.children[0], [&] { return exchange(false); });
            for (size_t edge = 0; edge + 1 < stages_.size(); ++edge)
            {
                const bool sender = index == edge, receiver = index == edge + 1;
                std::unique_ptr<IGPUGraphCapture> empty;
                auto &child = sender ? part.children[2] : receiver ? part.children[1] : empty;
                record(child, [&] {
                    if (!sender && !receiver) return true;
                    const CollectiveP2POp operation{
                        .kind = sender ? CollectiveP2POpKind::Send : CollectiveP2POpKind::Recv,
                        .send_buffer = sender ? stage.state_.hidden->gpu_data_ptr() : nullptr,
                        .recv_buffer = receiver ? stage.state_.hidden->gpu_data_ptr() : nullptr,
                        .count = static_cast<size_t>(part.width),
                        .dtype = CollectiveDataType::FLOAT32, .peer = static_cast<int>(sender ? edge + 1 : edge)};
                    return collective_->groupedP2PRawOnStream({operation}, static_cast<int>(index), part.stream,
                        "pipeline_forward_only_activation");
                });
            }
            const bool tail = index + 1 == stages_.size();
            record(part.children[3], [&] {
                // The existing backend publisher branches on ForwardOnly before
                // reading sampled_tokens, stop policy or history. No sampler node exists.
                return !tail || getBackendFor(part.device)->enqueuePublishOrdinaryGenerationSample(
                    part.publication, part.device.ordinal, part.stream);
            });
            record(part.children[4], [&] { return exchange(true); });
            if (failed.load()) return false;
            std::vector<GPUOrderedTimelineStep> steps{{.name = "original tail condition", .capture = part.children[0].get()}};
            if (index) steps.push_back({.name = "receive scalar activation", .capture = part.children[1].get()});
            steps.push_back({.name = "original resident model", .capture = part.forward->native()->capture});
            if (!tail) steps.push_back({.name = "send scalar activation", .capture = part.children[2].get()});
            else steps.push_back({.name = "forward-only state commit", .capture = part.children[3].get()});
            steps.push_back({.name = "tail completion", .capture = part.children[4].get()});
            for (const auto &fragment : stage.retainedStorageTerminalFragments())
                steps.push_back({.name = fragment.name, .capture = fragment.capture});
            part.parent = context->createGraphCapture(part.stream);
            if (!part.parent || !part.parent->buildOrderedTimelineTransaction(steps) || !part.parent->instantiate()) return false;
            PerfStatsCollector::addCounter("generation", "pipeline_forward_only_graph_materializations", 1.0,
                "graph_setup", part.device.toString(), {{"executable_owners", "1"}, {"sampler", "absent"}});
            return true;
        });
        if (!collectPreparation(*workers_)) { abortParticipants(); return false; }
        forwardOnlyFamily() = std::move(family);
        return forwardOnlyFamilyCurrent();
    };
    for (std::size_t program = 0; program < count; ++program)
    {
        std::vector<DeviceGraphOrchestrator::SparseQueryProgramCompilationScope> participants;
        participants.reserve(stages_.size());
        for (auto *stage : stages_)
            participants.emplace_back(*stage, program == 0 ? SparseQueryExecutionProgram::RankingPossible :
                SparseQueryExecutionProgram::ProvenNoRanking);
        if (!compile_program()) return false;
    }
    return true;
}

bool PipelineDeviceGeneration::prepareCapturedDomainForwardOnlyFamily()
{
    if (transport_ != Transport::CapturedDomains || !prefillPrepared()) return false;
    const auto programs = stages_.front()->sparseQueryProgramCount();
    for (auto *stage : stages_)
        if (stage->sparseQueryProgramCount() != programs || !stage->pipeline_forward_edges_ ||
            !stage->pipeline_forward_edges_->hasHeterogeneousBoundary()) return false;
    for (const auto &domain : domains_)
        if (domain.count > 1 && !dynamic_cast<RankOrchestrator *>(domain.runner))
            return false; // A mirrored domain must own its real rank metadata retirement.
    for (size_t program = 0; program < programs; ++program)
    {
        std::vector<DeviceGraphOrchestrator::SparseQueryProgramCompilationScope> scopes;
        scopes.reserve(stages_.size());
        for (auto *stage : stages_)
            scopes.emplace_back(*stage, program == 0 ? SparseQueryExecutionProgram::RankingPossible :
                SparseQueryExecutionProgram::ProvenNoRanking);
        if (forwardOnlyFamily())
        {
            if (!forwardOnlyFamilyCurrent()) return false;
            continue;
        }
        auto family = std::make_unique<ForwardOnlyFamily>();
        family->participants.resize(stages_.size());
        workers_->dispatch([&, this](size_t index)
        {
            auto &stage = *stages_[index];
            auto &part = family->participants[index];
            const auto &domain = domainFor(index);
            auto &gpu = GPUDeviceContextPool::instance().getContext(stage.state_.device_id);
            part.device = stage.state_.device_id;
            part.workspace = stage.workspaceGeneration(part.device);
            part.stream = stage.deviceGenerationLoopGraph().stream.get();
            part.arena = stage.arena_.get();
            part.domain_runner = domain.runner;
            part.domain_context = domain.context;
            part.domain_participant = index - domain.first;
            part.edges = stage.pipeline_forward_edges_;
            part.hidden = stage.state_.hidden->gpu_data_ptr();
            part.width = stage.graph_builder_->config().residualWidth();
            part.logits = stage.state_.logits->gpu_data_ptr();
            part.publication = ForwardOnlyFamily::publicationFor(stage);
            std::string error;
            // Serving retained this precise resident-token body. Its native
            // graph already contains live NextToken metadata and activation
            // channels, including the domain's actual TP collectives.
            part.forward = stage.materializeOrdinaryGenerationForward(
                stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device,
                OrdinaryForwardRetentionKind::NativeComposable, &error);
            stage.prepareRetainedStorageBinding();
            const auto storage = stage.retainedStorageTerminalFragments();
            part.storage_boundary = stage.retained_storage_boundary_.get();
            part.storage_terminal = stage.retained_storage_terminal_capture_.get();
            if (!part.workspace || !part.stream || !part.forward || !part.forward->native() ||
                part.width <= 0 || !part.publication.valid()) return false;
            std::vector<GPUOrderedTimelineStep> steps{{.name = "complete captured-domain resident forward",
                .capture = part.forward->native()->capture}};
            if (terminalParticipant(index))
            {
                // This recording has no collective and is imported, not
                // instantiated independently. ForwardOnly branches before all
                // sampler, response and generated-history reads/writes.
                part.children[3] = gpu.createGraphCapture(part.stream);
                if (!part.children[3]) return false;
                ScopedBackendGraphCapture recording(gpu, *part.children[3], "pipeline_domain_forward_only_publication");
                if (!recording.begin()) return false;
                const bool queued = getBackendFor(part.device)->enqueuePublishOrdinaryGenerationSample(
                    part.publication, part.device.ordinal, part.stream);
                recording.finish();
                if (!queued) return false;
                steps.push_back({.name = "forward-only terminal-domain state commit", .capture = part.children[3].get()});
            }
            for (const auto &fragment : storage)
                steps.push_back({.name = fragment.name, .capture = fragment.capture});
            part.parent = gpu.createGraphCapture(part.stream);
            if (!part.parent || !part.parent->buildOrderedTimelineTransaction(steps) || !part.parent->instantiate())
                return false;
            PerfStatsCollector::addCounter("generation", "pipeline_forward_only_graph_materializations", 1.0,
                "graph_setup", part.device.toString(), {{"executable_owners", "1"}, {"sampler", "absent"},
                    {"transport", "captured_domains"}});
            return true;
        });
        if (!collectPreparation(*workers_)) { abortParticipants(); return false; }
        forwardOnlyFamily() = std::move(family);
        if (!forwardOnlyFamilyCurrent()) return false;
    }
    return true;
}

bool PipelineDeviceGeneration::advanceOrdinaryCondition(
    const DeviceResidentLogicalSequenceStateHandle &logical_state, int request_index)
{
    for (const auto *stage : stages_)
        if (stage->usesParticipantLocalDeviceMoERebalanceController())
        {
            LOG_ERROR("Pipeline ForwardOnly does not admit a dynamic participant-local MoE maintenance policy");
            return false;
        }
    // Rank TP handles are aggregate ownership tokens, never device pointers.
    // Authenticate through the original terminal domain, then each DGO's own
    // admission consumes its current local mailbox on its exact event edge.
    if (phase_ != Phase::Idle || request_index != 0 || !forwardOnlyFamilyCurrent() ||
        !logical_state.coversRequest(0)) return false;
    const auto current = domains_.back().runner->deviceResidentLogicalSequenceState();
    if (!logical_state.sameMailboxAs(current)) return false;
    const DeviceGenerationAdmissionRequest admission{
        .request_count = 1, .max_new_tokens = 0,
        .depth_policy = sampling_math::DeviceGenerationPolicy::forwardOnly(),
        .initial_leading_row_disposition = sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted};
    if (!begin(admission)) return false;
    phase_ = Phase::Failed;
    // Join every old producer/observer before opening any replacement writer.
    // Observation correctly rejects an unfinished writer, even on the same stream.
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &part = forwardOnlyFamily()->participants[i];
        if (!stage.consumeDeviceGenerationStateReady(part.stream, DeviceTimelineRole::DeviceGenerationController,
                1, "pipeline_forward_only_parent") ||
            !stage.waitForLiveInferenceStateReadyForObservation(part.stream,
                "pipeline_forward_only_parent", DeviceTimelineRole::DeviceGenerationController) ||
            !stage.prepareMoEOverlayEpochForInternalParent(part.stream, "pipeline_forward_only_parent")) return false;
        if (stage.retained_storage_boundary_) stage.retained_storage_boundary_->requireArmedForLaunch();
    }
    // Submit every complete participant before observing any terminal result.
    // Once native launch accepts work, publication failure is fatal, never retryable.
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &part = forwardOnlyFamily()->participants[i];
        try
        {
            // The exclusive row lease closes through the actual post-launch event
            // before terminal observation, including a later failed terminal copy.
            if (!stage.waitForDeviceResidentLogicalSequenceStateRowReuse(
                    part.stream, "pipeline_forward_only_writer"))
            {
                LOG_ERROR("Pipeline ForwardOnly could not acquire its logical-row writer on " << part.device.toString());
                abortParticipants();
                std::terminate();
            }
            if (!part.parent->launchOnStream(part.stream))
            {
                LOG_ERROR("Pipeline ForwardOnly parent submission failed on " << part.device.toString());
                abortParticipants();
                std::terminate();
            }
            part.submitted = true;
            // Only the tail's real publisher produces the complete logical
            // frontier. A follower receives one condition field; its physical
            // access root must close without inventing the remaining fields.
            std::string publication_error;
            const bool logical_write_published = terminalParticipant(i)
                ? stage.recordDeviceResidentLogicalSequenceStateMailbox(1, part.stream,
                    DeviceGraphOrchestrator::DeviceResidentLogicalStatePublicationKind::ForwardOnlyTerminal,
                    &publication_error)
                : stage.publishPipelineFollowerLogicalRowWriteCompletion(part.stream);
            if (!logical_write_published)
            {
                LOG_ERROR("Pipeline ForwardOnly " << (i + 1 == stages_.size() ? "tail mailbox" : "follower write completion")
                    << " publication failed on " << part.device.toString() << ": " << publication_error);
                abortParticipants();
                std::terminate();
            }
            if (!stage.publishDeviceGenerationStateReady(part.stream, 1, DeviceGenerationStatePublicationKind::Terminal))
            {
                LOG_ERROR("Pipeline ForwardOnly terminal event publication failed on " << part.device.toString());
                abortParticipants();
                std::terminate();
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("Pipeline ForwardOnly submission/publication failed on " << part.device.toString() << ": " << error.what());
            abortParticipants();
            std::terminate();
        }
        catch (...)
        {
            LOG_ERROR("Pipeline ForwardOnly submission/publication failed on " << part.device.toString() << " with an unknown exception");
            abortParticipants();
            std::terminate();
        }
    }
    phase_ = Phase::Submitted;
    return finishForwardOnly();
}

bool PipelineDeviceGeneration::finishForwardOnly()
{
    using namespace sampling_math;
    if (phase_ != Phase::Submitted || !forwardOnlyFamilyCurrent()) return false;
    phase_ = Phase::Failed;
    auto &tail = *stages_[terminalIndex()];
    std::vector<std::array<int, 3>> terminal(stages_.size());
    std::vector<int32_t> published_cached_tokens(stages_.size(), -1);
    // These fixed terminal observations are local host backing. Every accepted
    // copy must retire before they leave scope, including false/throw exits.
    // Existing terminal events own the fence; a failed record/wait is fatal,
    // never permission to destroy storage with a possible native write in flight.
    const auto retire_copies = [&]() noexcept {
        try {
            for (size_t i = 0; i < stages_.size(); ++i)
            {
                const auto &part = forwardOnlyFamily()->participants[i];
                if (!part.terminal_copies_pending) continue;
                auto &stage = *stages_[i];
                auto *backend = getBackendFor(part.device);
                if (!backend || !stage.device_generation_terminal_host_ready_event_ ||
                    !backend->recordEvent(stage.device_generation_terminal_host_ready_event_.get(), part.device.ordinal, part.stream))
                    std::terminate();
            }
            for (size_t i = 0; i < stages_.size(); ++i)
            {
                auto &part = forwardOnlyFamily()->participants[i];
                if (!part.terminal_copies_pending) continue;
                if (!getBackendFor(part.device)->waitForEvent(
                        stages_[i]->device_generation_terminal_host_ready_event_.get(), part.device.ordinal))
                    std::terminate();
                part.terminal_copies_pending = false;
            }
        } catch (...) { std::terminate(); }
    };
    struct CopyRetirement
    {
        const decltype(retire_copies) &retire;
        /** @brief Retain every accepted terminal-copy destination through its exact completion. */
        ~CopyRetirement() noexcept { retire(); }
    } copy_retirement{retire_copies};
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        const auto &part = forwardOnlyFamily()->participants[i];
        auto *backend = getBackendFor(part.device);
        if (!stage.consumeDeviceGenerationStateReady(part.stream, DeviceTimelineRole::HostResultBridge,
                1, "pipeline_forward_only_terminal")) return false;
        forwardOnlyFamily()->participants[i].terminal_copies_pending = true;
        const std::array<const int *, 3> sources{
            stage.state_.kv_cache->deviceSequenceCachedTokenCountPtr(0),
            stage.device_generation_storage_.control_device + kDeviceGenerationControlOk,
            stage.device_generation_storage_.control_device + kDeviceGenerationControlRequestComplete};
        for (size_t field = 0; field < sources.size(); ++field)
            if (!backend->deviceToHostOnStream(&terminal[i][field], sources[field], sizeof(int),
                    part.device.ordinal, part.stream)) return false;
        if (terminalParticipant(i))
        {
            const auto host_controls = stage.deviceGenerationTerminalHostControls(1);
            if (host_controls.size() < static_cast<size_t>(kDeviceGenerationControlCount)) return false;
            // Fixed ABI metadata only. No response allocation capacity is data
            // for a zero-response operation, and no token/logit readback drives it.
            if (!backend->deviceToHostOnStream(&published_cached_tokens[i],
                    stage.device_resident_logical_sequence_state_storage_.target_cached_tokens_device,
                    sizeof(published_cached_tokens[i]), part.device.ordinal, part.stream) ||
                !backend->deviceToHostOnStream(host_controls.data(),
                    stage.device_generation_storage_.control_device, sizeof(int) * kDeviceGenerationControlCount,
                    part.device.ordinal, part.stream)) return false;
        }
    }
    retire_copies();
    for (auto *stage : stages_) stage->awaitRetainedStorageTerminal();
    // Join every terminal event before reading another participant's host result.
    for (size_t i = 0; i < terminal.size(); ++i)
    {
        const auto &local = terminal[i];
        // A captured-domain follower owns one input-row transaction, not a
        // response loop or full logical frontier. Its actual post-model event
        // completes the host submission; its own health must remain intact.
        const int expected_complete = transport_ == Transport::CapturedDomains && !terminalParticipant(i) ? 0 : 1;
        if (local[1] != 1 || local[2] != expected_complete || local[0] != terminal[terminalIndex()][0]) return false;
        if (!terminalParticipant(i)) continue;
        auto &stage = *stages_[i];
        const auto host_controls = stage.deviceGenerationTerminalHostControls(1);
        if (published_cached_tokens[i] != local[0] || !stage.active_device_generation_admission_ ||
            !stage.active_device_generation_admission_->depth_policy.isForwardOnly() ||
            validateDeviceGenerationTerminal(std::span<const int, kDeviceGenerationControlCount>(
                host_controls.data(), kDeviceGenerationControlCount),
                *stage.active_device_generation_admission_, stage.device_generation_storage_.response_token_stride,
                0, 0) != DeviceGenerationTerminalError::None) return false;
    }
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &part = forwardOnlyFamily()->participants[i];
        std::string error;
        if (!stage.forward_engine_->observeComposedForwardCompletion(part.forward->signature,
                part.forward->native()->capture, 1, &error) ||
            !stage.device_generation_state_ready_.handoff.retireAfterHostCompletion(1)) return false;
        stage.device_generation_storage_.active_request_count = 0;
        stage.active_device_generation_admission_.reset();
        part.submitted = false;
        stage.clearDeviceResidentLogicalSequenceStateMailbox();
    }
    for (size_t i = terminalIndex(); i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &part = forwardOnlyFamily()->participants[i];
        TransferEngine::publishDeviceWrite(stage.state_.logits.get(), stage.state_.device_id, part.stream);
        ForwardOutput output;
        output.logits = stage.state_.logits.get();
        output.hidden = stage.state_.hidden.get();
        output.execution = {.valid = true, .device = stage.state_.device_id, .stream = part.stream,
            .execution_role = ForwardExecutionRole::MainInference, .is_decode = true,
            .completion_scope = ForwardCompletionScope::ModelForwardOnly, .graph_seq_len = 1, .graph_batch_size = 1};
        if (!stage.publishForwardGraphOutputReady(output)) return false;
    }
    for (const auto &domain : domains_)
        if (auto *rank = dynamic_cast<RankOrchestrator *>(domain.runner))
            rank->retirePipelineForwardOnlyDomain();
    PerfStatsCollector::addCounter("generation", "pipeline_forward_only_completed_transactions", 1.0,
        "decode", "rank", {{"sparse_query_program", tail.sparseQueryForwardProgram(ForwardExecutionPhase::Decode) ==
            SparseQueryExecutionProgram::ProvenNoRanking ? "proven_no_ranking" : "ranking_possible"},
         {"sampler", "absent"}, {"response_tokens", "0"}, {"model_commits", "1"},
         {"transport", transport_ == Transport::CapturedDomains ? "captured_domains" : "native_pipeline"}});
    phase_ = Phase::Idle;
    return true;
}

PipelineDeviceGeneration::~PipelineDeviceGeneration()
{
    retireForwardOnlyBeforeStageRelease();
}

void PipelineDeviceGeneration::retireForwardOnlyBeforeStageRelease() noexcept
{
    workers_.reset(); // Join setup before any borrowed participant owner can retire.
    for (auto &family : forward_only_families_)
    {
        if (!family) continue;
        // A failed submitted transaction still owes its exact original event.
        for (size_t i = 0; i < stages_.size(); ++i)
            if (family->participants[i].submitted)
            {
                auto &stage = *stages_[i];
                auto *backend = getBackendFor(stage.state_.device_id);
                if (!backend || !stage.device_generation_state_ready_.event ||
                    !backend->waitForEvent(stage.device_generation_state_ready_.event.get(), stage.state_.device_id.ordinal))
                    std::terminate();
            }
        family.reset(); // Parent dies before its original collective/DGO sources.
    }
}

bool PipelineDeviceGeneration::isGPUStageSet(std::span<const std::unique_ptr<IInferenceRunner>> stages)
{
    if (stages.size() < 2) return false;
    return std::all_of(stages.begin(), stages.end(), [](const auto &stage) {
        return stage && stage->primaryDeviceId().is_gpu();
    });
}

const PipelineDeviceGeneration::Domain &PipelineDeviceGeneration::domainFor(size_t participant) const
{
    for (const auto &domain : domains_)
        if (participant >= domain.first && participant - domain.first < domain.count) return domain;
    throw std::out_of_range("Pipeline participant is outside its frozen domains");
}

void PipelineDeviceGeneration::abortParticipants()
{
    if (collective_) collective_->requestAbort();
    for (const auto &domain : domains_) if (domain.context) domain.context->requestAbort();
}

bool PipelineDeviceGeneration::supportsMTPPublication() const
{
    return prefillPrepared() && domains_.back().runner->supportsDeviceResidentMTPSpecStatePublication() &&
        std::all_of(stages_.begin(), stages_.begin() + terminalIndex(), [](const auto *stage) {
            return stage->supportsDeviceResidentMTPSpecStatePublication();
        });
}

void PipelineDeviceGeneration::prepareDomainBoundaries()
{
    for (size_t i = 1; i < domains_.size(); ++i)
    {
        auto &earlier = *stages_[domains_[i - 1].first];
        auto &later = *stages_[domains_[i].first];
        const auto &config = earlier.graph_builder_->config();
        // PMA may admit a smaller common resident family than the global
        // bucket ceiling. The initialized arena owns that final geometry;
        // re-reading startup policy here would allocate an unadmitted channel.
        const int admitted_rows = earlier.state_.activation_seq_len;
        if (admitted_rows <= 0 || later.state_.activation_seq_len != admitted_rows)
            throw std::logic_error("Captured pipeline boundary requires one common admitted activation capacity");
        const auto geometry = PipelineTransferMemory::forRows(config.residualWidth(),
            admitted_rows, std::max(1, resolveMTPRetainedTargetQueryRows(config.mtp)));
        auto &transfer = TransferEngine::instance();
        const auto create = [&](auto &source, auto &destination, size_t capacity) {
            return transfer.createCapturedTransferChannel(*earlier.physical_memory_authority_, DeviceId::cpu(),
                source.state_.device_id, source.deviceGenerationLoopGraph().stream.get(),
                destination.state_.device_id, destination.deviceGenerationLoopGraph().stream.get(), capacity);
        };
        boundaries_.push_back({create(earlier, later, geometry.activation_capacity_bytes),
            create(later, earlier, geometry.metadata_capacity_bytes)});
    }
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
        if (collective_ && !collective_->reserveGraphCaptureBoundaryResources(memory)) return false;
        // A late edge install would change already captured topology. Reject
        // it before mutating any stage instead of invalidating/recapturing a
        // supposedly ready model. The owner is installed at serving setup.
        for (auto *stage : stages_)
            if (stage->forward_engine_ || stage->pipeline_forward_edges_)
                throw std::logic_error("Pipeline prefill transport must be frozen before forward-engine construction");
        if (transport_ == Transport::CapturedDomains)
        {
            // A cross-backend edge retains the arena's exact physical bank.
            // Ordinary activations otherwise acquire storage at their first
            // graph preparation; materialize this admitted bank now, before
            // freezing any channel binding. This neither copies host data nor
            // creates a second allocation owner or a mutable staging buffer.
            for (auto *stage : stages_)
                if (!stage->arena_->allocateDeviceStorage(BufferId::HIDDEN_STATE, stage->state_.device_id))
                    throw std::runtime_error("Pipeline activation bank could not materialize its admitted device storage");
            prepareDomainBoundaries();
        }
        for (size_t index = 0; index < stages_.size(); ++index)
        {
            auto &stage = *stages_[index];
            const auto &domain = domainFor(index);
            const auto domain_index = size_t(&domain - domains_.data());
            std::optional<PipelineForwardGraphEdges::FollowerState> follower;
            if (!terminalParticipant(index) && stage.graph_builder_->config().mtp.enabled)
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
            if (transport_ == Transport::NativePipeline)
                forward_edges_.push_back(std::make_unique<PipelineForwardGraphEdges>(*collective_,
                    static_cast<int>(index), *stage.state_.hidden, stage.graph_builder_->config().residualWidth(), follower));
            else
                forward_edges_.push_back(std::make_unique<PipelineForwardGraphEdges>(stage.state_.device_id,
                    PipelineForwardGraphEdges::CapturedDomain{
                        .stage_index = domain_index, .stage_count = domains_.size(), .context = domain.context,
                        .participant = int(index - domain.first),
                        .activation_in = domain_index ? boundaries_[domain_index - 1].activation : nullptr,
                        .activation_out = domain_index < boundaries_.size() ? boundaries_[domain_index].activation : nullptr,
                        .metadata_in = domain_index < boundaries_.size() ? boundaries_[domain_index].metadata : nullptr,
                        .metadata_out = domain_index ? boundaries_[domain_index - 1].metadata : nullptr},
                    *stage.arena_, stage.graph_builder_->config().residualWidth(), follower));
            stage.pipeline_forward_edges_ = forward_edges_.back().get();
        }
    }
    workers_->dispatch([&, this](size_t index) {
        const auto &domain = domainFor(index);
        if (index != domain.first) return true; // The TP runner owns its own symmetric worker wave.
        auto local = plan;
        // The receive node fills the stage's own bank. No external tensor is
        // migrated or aliased into this participant's captured model graph.
        local.pipeline_hidden_input = index ? stages_[index]->state_.hidden.get() : nullptr;
        return domain.runner->materializeServingGraphFamilyWithoutLaunch(local);
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
        std::vector<PipelineMetadataBank> retained;
        if (transport_ == Transport::CapturedDomains)
        {
            auto *workspace = stage.workspace_allocator_->getDeviceWorkspace(stage.state_.device_id);
            if (!workspace) return false;
            auto accepted = workspace->retainBuffer(MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES,
                0, sizeof(int32_t));
            if (accepted->data() != metadata.accepted_state_slot_indices)
                throw std::logic_error("Pipeline publication restore bank lost its original workspace owner");
            retained.emplace_back(std::move(accepted), 0, 1);
            auto logical = stage.arena_->getSharedTensor(BufferId::MTP_LOGICAL_SEQUENCE_STATE);
            if (!logical || !logical->gpu_data_ptr()) return false;
            const auto base = reinterpret_cast<uintptr_t>(logical->gpu_data_ptr());
            for (const auto *field : {metadata.target_cached_tokens, metadata.accepted_state_counts, metadata.publication_ok_flags})
            {
                const auto address = reinterpret_cast<uintptr_t>(field);
                if (address < base) throw std::logic_error("Pipeline publication metadata precedes its logical-state arena");
                retained.emplace_back(logical, BufferId::MTP_LOGICAL_SEQUENCE_STATE, address - base, 1);
            }
        }
        stage.pipeline_publication_transport_ = forward_edges_[index]->materializePublicationTransport(
            {metadata.accepted_state_slot_indices, metadata.target_cached_tokens,
                metadata.accepted_state_counts, metadata.publication_ok_flags}, stage.executor_, *context, gpu,
                std::move(retained));
        return static_cast<bool>(stage.pipeline_publication_transport_);
    });
    if (!collectPreparation(*workers_)) return false;
    if (!prepareForwardOnlyFamily()) return false;
    phase_ = Phase::Idle;
    return true;
}

bool PipelineDeviceGeneration::prefill(const int *tokens, int count,
    const PrefillChunkSchedulerPolicy &policy, int pad_token, bool allow_padding)
{
    if (phase_ != Phase::Idle || !prefillPrepared() || !tokens || count <= 0) return false;
    phase_ = Phase::Failed;
    workers_->dispatch([&, this](size_t index) {
        const auto &domain = domainFor(index);
        return index != domain.first || domain.runner->forwardPrefillChunkSchedule(tokens, count, policy, pad_token, allow_padding);
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

bool PipelineDeviceGeneration::forwardDecodeInput(int32_t token_id)
{
    const int token = token_id;
    return forwardMainInput([&](IInferenceRunner &runner) { return runner.forward(&token, 1); });
}

bool PipelineDeviceGeneration::restoredPrefixMTPDecodeBridge(
    const RestoredPrefixMTPDecodeBridgeRequest &request)
{
    return request.valid() && forwardMainInput([&](IInferenceRunner &runner) {
        return runner.forwardRestoredPrefixMTPDecodeBridge(request);
    });
}

bool PipelineDeviceGeneration::forwardMainInput(
    const std::function<bool(IInferenceRunner &)> &forward)
{
    if (phase_ != Phase::Idle || !prefillPrepared() || !forward) return false;
    phase_ = Phase::Failed;
    // Each domain's existing forward API owns TP admission and exact history
    // validation. Native activation edges, not the host's stage loop, deliver
    // intermediate rows; every sender and receiver must therefore be admitted.
    workers_->dispatch([&, this](size_t index) {
        const auto &domain = domainFor(index);
        if (index != domain.first) return true;
        if (!forward(*domain.runner)) return false;
        // As with an ordinary pipeline MTP condition, upstream domains have
        // no sampler to consume a main-logits handoff. Their activation edge
        // has already published the real output. Retire the unused deferral
        // before the next condition graph publishes on that same participant.
        if (!terminalParticipant(index))
            for (size_t member = domain.first; member < domain.first + domain.count; ++member)
                stages_[member]->setMTPMainDecodeSyncDeferralEnabled(false);
        return true;
    });
    if (!collectPreparation(*workers_)) return false;
    phase_ = Phase::Idle;
    return true;
}

bool PipelineDeviceGeneration::forwardMTPVerifier(int logical_rows,
    const std::function<bool(IInferenceRunner &)> &terminal)
{
    auto &tail = *stages_[terminalIndex()];
    std::string error;
    const auto width_policy = tail.mtpVerifierPhysicalWidthPolicyForRequest(1, &error);
    const auto geometry = resolveMTPServingForwardCaptureGeometry(tail.graph_builder_->config().mtp,
        tail.mtp_max_verifier_rows_, tail.mtpVerifierGraphFamily());
    const int physical_rows = width_policy ? mtpVerifierPhysicalPaddedSeqLen(
        1, logical_rows, tail.mtp_max_verifier_rows_, *width_policy) : 0;
    if (!geometry.enabled || !geometry.valid() || logical_rows < 2 || physical_rows < logical_rows ||
        physical_rows > geometry.verifier_rows ||
        !mtpVerifierGraphFamilyContains(physical_rows, geometry.verifier_rows, geometry.family))
    {
        LOG_ERROR("Pipeline verifier is not a retained graph member: logical_rows=" << logical_rows
            << " physical_rows=" << physical_rows << " retained_rows=" << geometry.verifier_rows << " " << error);
        return false;
    }
    // The callback installs the tail's greedy/stochastic outcome policy.
    // Followers only produce layer state and activations; asking them to arm
    // an outcome transaction would invent a second sampler/head authority.
    return forwardMTP(MTPVerifierForwardPolicy(MTPVerifierOutcomeGraphMode::Disabled, physical_rows, geometry), terminal);
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
        if (terminalParticipant(index))
            return index != terminalIndex() || terminal(*domains_.back().runner);
        std::string error;
        void *const stream = stage.explicitGPUStreamForOperation("pipeline_mtp_main_forward");
        // This stream publishes inputs; the retained verifier may execute on
        // another stream. Its live-state prelude must borrow admission on that
        // actual consumer so restored KV/GDN writes precede every graph root.
        const bool ok = stage.executeMTPMainForward(policy,
            domainFor(index).first ? stage.state_.hidden.get() : nullptr,
            stream,
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
    if (phase_ != Phase::Idle || !request.valid() || request.request_count != 1)
        return false;
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        const auto &stage = *stages_[i];
        const auto &domain = domainFor(i);
        if (!stage.pp_stage_config_ ||
            stage.pp_stage_config_->has_embedding != (domain.first == 0) ||
            stage.pp_stage_config_->has_lm_head != terminalParticipant(i) ||
            (collective_ && stage.state_.device_id != collective_->devices()[i].toLocalDeviceId()))
            throw std::logic_error("Pipeline generation stage ownership changed after collective admission");
        (void)PipelineNativeDomain(stage.state_.device_id, domain.context, int(i - domain.first));
        if(!qwen4exp::QwenSparseWorkCounterRow{stage.device_generation_storage_.controlForRequest(0),
                sampling_math::kDeviceGenerationControlCount}.valid())
            throw std::logic_error("Pipeline follower reporting span lost its canonical controller owner/alignment");
    }
    // Partial admission must never re-enter an unrelated rank/TP path. Until
    // every participant succeeds, this owner remains in an absorbing failure
    // state; the caller must retire the failed runner, not retry its request.
    phase_ = Phase::Failed;
    // The tail is the only owner permitted to initialize a sampler, response
    // budget or logical next-token authority. Followers admit local work only.
    if (!domains_.back().runner->beginDeviceResidentGeneration(request)) return false;
    for (size_t i = 0; i < terminalIndex(); ++i)
    {
        auto &stage = *stages_[i];
        void *stream = stage.explicitGPUStreamForOperation("pipeline_follower_admission");
        if (!stream || stage.device_generation_storage_.active_request_count != 0 ||
            stage.active_device_generation_admission_ || stage.deviceGenerationLoopGraph().launched ||
            !stage.device_generation_state_ready_.handoff.inactive() ||
            !stage.waitForPendingRequestStateReset(stream, DeviceTimelineRole::RequestAdmissionTransfer,
                "pipeline_follower_admission") ||
            !stage.waitForLiveInferenceStateReadyForObservation(stream,
                "pipeline_follower_admission", DeviceTimelineRole::DeviceGenerationController))
            return false;
        // Reporting is per response admission. Reset only the original four
        // pairs after old readers/reset have retired; received command words
        // remain owned by their existing arrival protocol.
        auto* backend=getBackendFor(stage.state_.device_id);
        if(!backend || !resetQwenSparseFollowerCounters(
                {stage.device_generation_storage_.controlForRequest(0),sampling_math::kDeviceGenerationControlCount},
                [&](void* span,std::size_t bytes){return backend->memset(span,0,bytes,stage.state_.device_id.ordinal,stream);}))
            return false;
        if (request.depth_policy.isForwardOnly())
        {
            // Unlike a response loop, the fixed one-shot does not overwrite
            // follower health with a tail command. Initialize its own original
            // control with the model-owned sticky-forward failure policy.
            if (!backend->enqueueInitializeDeviceGeneration(1, 0, request.depth_policy,
                    request.initial_leading_row_disposition,
                    stage.device_generation_storage_.response_token_stride,
                    stage.device_generation_storage_.response_tokens_device,
                    stage.device_generation_storage_.control_stride,
                    stage.device_generation_storage_.control_device, stage.state_.device_id.ordinal, stream,
                    {.kind = request.kind, .session_epoch = stage.session_epoch_,
                     .workspace_generation = stage.workspaceGeneration(stage.state_.device_id),
                     .prior_tickets = stage.device_generation_storage_.dispatch_tickets_device,
                     .forward_failure_policy = stage.graph_builder_->deviceGenerationForwardFailurePolicy()})) return false;
        }
        stage.admitSparseQueryProgram(request);
        stage.device_generation_storage_.active_request_count = 1;
        if(!stage.beginRetainedStorageAdmission(stream))return false;
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
        if (terminalParticipant(index))
        {
            if (index != terminalIndex()) return true;
            if (transport_ == Transport::CapturedDomains)
                return domains_.back().runner->materializeDeviceResidentGeneration(1, 0,
                    DeviceGenerationLoopTopology::FixedDepth, sampling);
            // Native PP owns the complete parent, including activation and
            // command edges. The tail's overlay reader lease remains inside
            // its retained forward; its no-peer proof is checked by the
            // materializer before it records either sampler child.
            const auto composition = stage.moe_overlay_epoch_execution_binding_
                ? DeviceGraphOrchestrator::OrdinaryGenerationComposition::PipelineNoPeerOverlayTail
                : DeviceGraphOrchestrator::OrdinaryGenerationComposition::PipelineTail;
            const bool ready = stage.materializeOrdinaryDeviceGenerationLoopGraph(sampling, &error, composition);
            if (!ready) LOG_ERROR("Pipeline tail preparation failed: " << error);
            return ready;
        }
        const auto forward = stage.materializeOrdinaryGenerationForward(
            stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device,
            OrdinaryForwardRetentionKind::NativeComposable, &error);
        if (!forward || !forward->native()) { LOG_ERROR("Pipeline follower requires a native-composable forward: " << error); return false; }
        auto &loop = stage.deviceGenerationLoopGraph();
        const auto generation = stage.workspaceGeneration(stage.state_.device_id);
        if (loop.valid && loop.workspace_generation == generation &&
            loop.ordinary_forward_identity == forward->signature && loop.source_fragments.size() == 1 &&
            loop.source_fragments.front().capture == forward->native()->capture)
            return true;
        loop.invalidateGraph();
        loop.ordinary_forward_identity = forward->signature;
        loop.source_fragments.push_back({"pipeline_local_forward", forward->native()->capture});
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
    if (transport_ == Transport::CapturedDomains)
    {
        // Forward captures already contain their declared boundary. Compose
        // only participant-local nodes; the terminal domain's existing compiler
        // already owns its sampler and authenticated ticket publisher.
        workers_->dispatch([this](size_t index) {
            if (terminalParticipant(index)) return true;
            auto &loop = stages_[index]->deviceGenerationLoopGraph();
            loop.capture->reset();
            const GPUOrderedTimelineStep forward{.name = "complete pipeline domain forward",
                .capture = loop.source_fragments.front().capture};
            return loop.capture->buildOrderedTimelineTransaction(std::span(&forward, 1)) && loop.capture->instantiate();
        });
        return collectPreparation(*workers_);
    }
    using namespace sampling_math;
    constexpr size_t command = 0, receive = 1, send = 2;
    const bool hosted = execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    std::barrier boundary(static_cast<ptrdiff_t>(stages_.size()));
    std::atomic<bool> failed{false};
    workers_->dispatch([&, this](size_t index) {
        auto &stage = *stages_[index];
        auto &loop = stage.deviceGenerationLoopGraph();
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
                    .count = static_cast<size_t>(stage.graph_builder_->config().residualWidth()),
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
            (void)stage.retainedStorageTerminalFragments();
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
                        stage.device_generation_storage_.dispatch_tickets_device, detail,stage.retained_storage_boundary_.get()))
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
                .iteration = body,.terminal=stage.retainedStorageTerminalFragments()};
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

bool PipelineDeviceGeneration::prepareContinuation(DeviceGenerationSamplingMode sampling)
{
    using namespace sampling_math;
    const auto &admission = stages_[terminalIndex()]->active_device_generation_admission_;
    const auto reject = [&](const std::string &reason) {
        LOG_ERROR("Pipeline retained continuation preparation rejected: " << reason
            << " phase=" << static_cast<int>(phase_) << " execution=" << static_cast<int>(execution_policy_)
            << " tail_admission=" << admission.has_value()
            << " admission_kind=" << (admission ? static_cast<int>(admission->kind) : -1)
            << " depth_policy=" << (admission ? static_cast<int>(admission->depth_policy.mode) : -1)
            << " sampling=" << static_cast<int>(sampling));
        return false;
    };
    if (phase_ != Phase::Admitted || execution_policy_ != DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions ||
        !admission || admission->kind != DeviceGenerationAdmissionKind::ContinueResponse ||
        admission->depth_policy.mode != DeviceGenerationPolicyMode::Dynamic || !isValidDeviceGenerationSamplingMode(sampling))
        return reject("requires one admitted Dynamic ContinueResponse with the declared hosted policy");
    phase_ = Phase::Failed;
    const auto count = stages_.front()->sparseQueryProgramCount();
    for (std::size_t program = 0; program < count; ++program)
    {
        std::vector<DeviceGraphOrchestrator::SparseQueryProgramCompilationScope> scopes;
        scopes.reserve(stages_.size());
        for (auto *stage : stages_)
        {
            if (stage->sparseQueryProgramCount() != count)
                return reject("participants disagree on their retained query-program family");
            scopes.emplace_back(*stage, program == 0 ? SparseQueryExecutionProgram::RankingPossible :
                SparseQueryExecutionProgram::ProvenNoRanking);
        }
        // Arm the tail once: its preparation authenticates every query program.
        // Followers own only exact verifier parents, never controller shadows.
        workers_->dispatch([&, this](size_t index) {
            if (terminalParticipant(index)) return true;
            const auto &stage = *stages_[index];
            const auto &loop = stage.deviceGenerationLoopGraph();
            const auto reject_member = [&](const std::string &reason, int rows = -1) {
                LOG_ERROR("Pipeline retained continuation member rejected: " << reason
                    << " participant=" << index << " device=" << stage.state_.device_id.toString()
                    << " program=" << sparseQueryExecutionProgramIndex(stage.sparseQueryGraphProgram())
                    << " rows=" << rows << " valid=" << loop.valid << " launched=" << loop.launched
                    << " requests=" << loop.request_count << " execution=" << static_cast<int>(loop.execution_kind)
                    << " workspace=" << loop.workspace_generation
                    << " expected_workspace=" << stage.workspaceGeneration(stage.state_.device_id)
                    << " depth_range=[" << loop.minimum_draft_depth << ',' << loop.maximum_draft_depth << ']'
                    << " expected_depth_range=[" << admission->depth_policy.minimum_depth << ','
                    << admission->depth_policy.maximum_depth << ']' << " retained_widths=" << loop.pipeline_verifier_branches.size());
                return false;
            };
            if (!loop.valid || loop.launched || !loop.stream || loop.request_count != 1 ||
                stage.mtpVerifierGraphFamily() != MTPVerifierGraphFamily::ExactRows ||
                loop.execution_kind != DeviceGraphOrchestrator::MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedPipelineTransaction ||
                loop.workspace_generation != stage.workspaceGeneration(stage.state_.device_id) ||
                loop.minimum_draft_depth != admission->depth_policy.minimum_depth ||
                loop.maximum_draft_depth != admission->depth_policy.maximum_depth ||
                loop.pipeline_verifier_branches.size() != mtpVerifierGraphFamilyWidthCount(
                    stage.mtp_max_verifier_rows_, stage.mtpVerifierGraphFamily()))
                return reject_member("retained follower family ownership, lifecycle, or immutable admission bounds disagree");
            for (std::size_t width = 0; width < loop.pipeline_verifier_branches.size(); ++width)
            {
                const auto &branch = loop.pipeline_verifier_branches[width];
                const int rows = mtpVerifierGraphFamilyWidthAt(width, stage.mtp_max_verifier_rows_, stage.mtpVerifierGraphFamily());
                if (!branch.capture || !branch.capture->hasExecutable() || branch.verifier_signature !=
                    stage.retainedMTPVerifierSignature(rows, MTPVerifierOutcomeGraphMode::Disabled,
                        stage.sparseQueryGraphProgram()))
                    return reject_member("complete follower parent has no current exact verifier signature/executable", rows);
                std::string error;
                const auto forward = stage.forward_engine_->deviceLoopGraphTemplate(branch.verifier_signature, &error);
                if (!forward) return reject_member("exact verifier source export failed: " + error, rows);
                const auto publication = stage.mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(1, rows, &error);
                if (!publication) return reject_member("accepted-state source export failed: " + error, rows);
                const bool epoch = static_cast<bool>(stage.moe_overlay_epoch_execution_binding_);
                const std::size_t offset = epoch ? 1u : 0u;
                if (branch.source_fragments.size() != (epoch ? 4u : 2u) ||
                    branch.source_fragments[offset].capture != forward->capture ||
                    branch.source_fragments[offset + 1].capture != publication->capture)
                    return reject_member("complete follower parent source recording identity changed", rows);
                if (epoch)
                {
                    const auto acquire = stage.moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
                        MoEOverlayEpochBoundaryStage::Operation::Acquire, &error);
                    if (!acquire) return reject_member("epoch acquire source export failed: " + error, rows);
                    const auto release = stage.moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
                        MoEOverlayEpochBoundaryStage::Operation::Release, &error);
                    if (!release) return reject_member("epoch release source export failed: " + error, rows);
                    if (branch.source_fragments.front().capture != acquire->capture ||
                        branch.source_fragments.back().capture != release->capture)
                        return reject_member("complete follower parent epoch source recording identity changed", rows);
                }
            }
            return true;
        });
        if (!collectPreparation(*workers_)) return reject("at least one retained follower family failed authentication");
    }
    if (!domains_.back().runner->prepareDeviceResidentGenerationContinuation(sampling))
        return reject("terminal domain rejected retained-family authentication or native admission-ticket preparation");
    phase_ = Phase::Materialized;
    return true;
}

bool PipelineDeviceGeneration::materialize(int requests, int depth,
    DeviceGenerationLoopTopology topology, DeviceGenerationSamplingMode sampling)
{
    if (phase_ != Phase::Admitted || requests != 1 || depth < 0 ||
        !isValidDeviceGenerationSamplingMode(sampling))
        return false;
    const auto count = stages_.front()->sparseQueryProgramCount();
    if (std::any_of(stages_.begin(), stages_.end(), [count](auto *stage) {
            return stage->sparseQueryProgramCount() != count; }))
        throw std::logic_error("Pipeline sparse-query programs disagree with the original model geometry");
    const auto compile_program = [&]() -> bool
    {
        phase_ = Phase::Failed;
        if (depth > 0)
        {
            if (!prepareSpeculativeParticipants(depth, topology, sampling)) return false;
            const bool reusable = std::all_of(stages_.begin(), stages_.end(), [this](auto *stage) {
                const auto &loop = stage->deviceGenerationLoopGraph();
                return loop.capture->hasExecutable() &&
                    (execution_policy_ != DeviceGenerationExecutionPolicy::NativeConditionalGraph ||
                     loop.pipeline_children[0]);
            });
            if (!reusable && !composeSpeculativeParticipants()) return false;
            phase_ = Phase::Materialized;
            return true;
        }
        if (topology != DeviceGenerationLoopTopology::FixedDepth) return false;
        if (!prepareParticipants(sampling)) return false;
        const bool reusable = std::all_of(stages_.begin(), stages_.end(), [](auto *stage) {
            return stage->deviceGenerationLoopGraph().capture->hasExecutable(); });
        if (!reusable && !composeParticipants()) return false;
        phase_ = Phase::Materialized;
        return true;
    };
    for (std::size_t program = 0; program < count; ++program)
    {
        std::vector<DeviceGraphOrchestrator::SparseQueryProgramCompilationScope> participants;
        participants.reserve(stages_.size());
        for (auto *stage : stages_)
            participants.emplace_back(*stage, program == 0 ? SparseQueryExecutionProgram::RankingPossible :
                SparseQueryExecutionProgram::ProvenNoRanking);
        if (!compile_program()) return false;
    }
    return true;
}

bool PipelineDeviceGeneration::launch()
{
    if (phase_ != Phase::Materialized) return false;
    phase_ = Phase::Failed;
    if (execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)
    {
        const bool speculative = stages_[terminalIndex()]->deviceGenerationLoopGraph().draft_depth > 0;
        if (!(speculative ? launchHostedSpeculativeTransactions() : launchHostedTransactions()))
        {
            abortParticipants();
            throw std::runtime_error("Pipeline generation failed during ticket-selected distributed submission");
        }
        phase_ = Phase::Submitted;
        return true;
    }
    for (auto *stage : stages_)
        if (!stage->launchDeviceResidentGeneration())
        {
            abortParticipants();
            throw std::runtime_error("Pipeline generation failed during distributed submission");
        }
    phase_ = Phase::Submitted;
    return true;
}

bool PipelineDeviceGeneration::launchHostedTransactions()
{
    if (transport_ == Transport::CapturedDomains) return launchHostedDomainTransactions();
    auto &tail = *stages_.back();
    auto &tail_loop = tail.deviceGenerationLoopGraph();
    for (auto *stage : stages_)
    {
        auto &loop = stage->deviceGenerationLoopGraph();
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
            auto &loop = stages_[i]->deviceGenerationLoopGraph();
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
                if (!stages_[i]->submitRetainedStorageTerminal()||!stages_[i]->publishDeviceGenerationStateReady(
                        stages_[i]->deviceGenerationLoopGraph().stream.get(), 1,
                        DeviceGenerationStatePublicationKind::Terminal)) return false;
            return true;
        }
    }
    throw std::runtime_error("Pipeline generation exhausted its admitted response bound without a terminal ticket");
}

bool PipelineDeviceGeneration::launchHostedDomainTransactions()
{
    auto &tail = *stages_[terminalIndex()];
    auto &owner = *domains_.back().runner;
    for (size_t i = 0; i < terminalIndex(); ++i)
    {
        auto &stage = *stages_[i];
        auto &loop = stage.deviceGenerationLoopGraph();
        if (!loop.valid || loop.launched || !loop.capture->hasExecutable() ||
            loop.workspace_generation != stage.workspaceGeneration(stage.state_.device_id) ||
            !stage.consumeDeviceGenerationStateReady(loop.stream.get(),
                DeviceTimelineRole::DeviceGenerationController, 1, "pipeline_domain_admission")) return false;
        loop.launched = true;
    }
    const auto submit_followers = [&] {
        // Every participant is enqueued before the terminal is allowed to wait
        // for a ticket. Native broadcasts remain on their exact device streams.
        for (size_t i = 0; i < terminalIndex(); ++i)
        {
            auto &loop = stages_[i]->deviceGenerationLoopGraph();
            if (!loop.capture->launchOnStream(loop.stream.get())) return false;
        }
        return true;
    };
    if (tail.deviceGenerationLoopGraph().ordinary_leading ==
            sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted && !submit_followers())
        return false;
    const int bound = tail.active_device_generation_admission_->max_new_tokens;
    for (int observation = 0; observation < bound; ++observation)
    {
        sampling_math::DeviceGenerationDispatchTicket ticket;
        if (!owner.observeDeviceGenerationDispatchTicket(&ticket) ||
            (!ticket.complete && !submit_followers()) ||
            !owner.submitHostScheduledDeviceGenerationAdvance(ticket)) return false;
        if (!ticket.complete) continue;
        for (size_t i = 0; i < terminalIndex(); ++i)
            if (!stages_[i]->submitRetainedStorageTerminal()||!stages_[i]->publishDeviceGenerationStateReady(stages_[i]->deviceGenerationLoopGraph().stream.get(),
                    1, DeviceGenerationStatePublicationKind::Terminal)) return false;
        return true;
    }
    throw std::runtime_error("Pipeline domains exhausted their admitted response bound without a terminal ticket");
}

bool PipelineDeviceGeneration::finish(DeviceGenerationTerminalResult *result)
{
    if (phase_ != Phase::Submitted || !result) return false;
    phase_ = Phase::Failed;
    if (!domains_.back().runner->finishDeviceResidentGeneration(result)) return false;
    struct TerminalObservation {
        std::array<int,3> state{}; ///< Existing independent KV/health terminal evidence.
        QwenSparseFollowerCounterObservation counters; ///< Exact final32B, never execution state.
    };
    std::vector<TerminalObservation> terminal(stages_.size());
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto *backend = getBackendFor(stage.state_.device_id);
        void *stream = stage.deviceGenerationLoopGraph().stream.get();
        const bool follower = !terminalParticipant(i);
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
            if (!backend->deviceToHostOnStream(&terminal[i].state[field], sources[field], sizeof(int),
                    stage.state_.device_id.gpu_ordinal(), stream)) return false;
        // Extend this participant's existing final observation, on the same
        // terminal-ready stream/event. Tail/domain-tail counters are already
        // projected by their DGO terminal bridge and must never be duplicated.
        if(!observeQwenSparseFollowerCounters(
                follower ? QwenSparseTerminalOwner::PipelineFollower : QwenSparseTerminalOwner::DeviceControllerTail,
                {stage.device_generation_storage_.controlForRequest(0),sampling_math::kDeviceGenerationControlCount},
                terminal[i].counters,[&](void* destination,const void* source,std::size_t bytes){
                    return backend->deviceToHostOnStream(destination,source,bytes,stage.state_.device_id.ordinal,stream);}))
            return false;
        if (!backend->recordEvent(stage.device_generation_terminal_host_ready_event_.get(),
                stage.state_.device_id.gpu_ordinal(), stream)) return false;
    }
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        const bool hosted_follower = !terminalParticipant(i) &&
            execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
        if (!getBackendFor(stage.state_.device_id)->waitForEvent(stage.device_generation_terminal_host_ready_event_.get(),
                stage.state_.device_id.gpu_ordinal()) || (!hosted_follower && (terminal[i].state[1] != 1 || terminal[i].state[2] != 1)))
            return false;
        if (!terminalParticipant(i))
        {
            stage.awaitRetainedStorageTerminal();
            if (!stage.deviceGenerationLoopGraph().retireCompletedLaunch() ||
                !stage.device_generation_state_ready_.handoff.retireAfterHostCompletion(1)) return false;
            stage.device_generation_storage_.active_request_count = 0;
        }
    }
    for (const auto &local : terminal)
        if (local.state[0] != terminal.back().state[0])
            throw std::runtime_error("Pipeline generation terminal KV positions disagree; no state repair is permitted");
    // The tail reported its own completed forwards through the ordinary DGO
    // terminal. Followers report only after the independent KV check above;
    // they borrow its authenticated count, never a mirrored response ledger.
    for (size_t i = 0; i < terminalIndex(); ++i)
        if (stages_[i]->deviceGenerationLoopGraph().draft_depth == 0 &&
            !stages_[i]->observeOrdinaryGenerationForwardCompletion(
                result->requests.front().published_state_commit_count)) return false;
    // Every participant's complete event, health and canonical KV position is
    // validated above before any new follower observation becomes PerfStats.
    const char* names[]{"verifier_prior_header_bytes","verifier_prior_payload_bytes",
        "verifier_candidate_payload_bytes","verifier_accepted_payload_bytes"};
    for(std::size_t participant=0;participant<terminalIndex();++participant){
        const auto total=terminal[participant].counters.totals();
        const std::uint64_t bytes[]{total.prior_header,total.prior_payload,total.candidate_payload,total.accepted_payload};
        for(int metric=0;metric<4;++metric)
            PerfStatsCollector::addCounterWithSequence("qsa",names[metric],static_cast<double>(bytes[metric]),
                {bytes[metric]},"decode",stages_[participant]->state_.device_id.toString(),
                {{"authority","pipeline_follower_terminal"},{"request","0"}});
        PerfStatsCollector::addCounter("qsa","verifier_follower_terminal_observation_bytes",32.0,"decode",
            stages_[participant]->state_.device_id.toString(),{{"authority","pipeline_follower_terminal"}});
    }
    PerfStatsCollector::addCounter("generation", "pipeline_terminal_responses", 1.0, "decode", "rank",
        {{"response_owner", "tail"}, {"validation", "independent_stage_kv_positions"}});
    phase_ = Phase::Idle;
    return true;
}
}
