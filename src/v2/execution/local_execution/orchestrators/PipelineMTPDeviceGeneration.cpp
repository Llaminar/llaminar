/**
 * @file PipelineMTPDeviceGeneration.cpp
 * @brief Compose speculative pipeline generation around one terminal authority.
 *
 * The tail owns drafts, sampling, depth selection and the response budget.
 * Followers own only their layer verifier and accepted-state publication. CUDA
 * receives a continuation command inside a complete native parent; HIP submits
 * one complete follower transaction from the tail's authenticated scheduler
 * ticket. Neither path copies mutable model or controller state to the host.
 * Each sparse follower retains its own device epoch acquire/release around
 * verification and publication; the tail's lease covers only its own domain.
 */
#include "PipelineDeviceGeneration.h"
#include "DeviceGraphOrchestrator.h"
#include "TPWorkerPool.h"
#include "backends/BackendManager.h"
#include "collective/LocalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <stdexcept>

namespace llaminar2
{
namespace
{
/** @brief Join setup submissions without observing any device-owned decision. */
bool joinPipelineMTP(TPWorkerPool &workers)
{
    for (const auto &result : workers.collectAll(
             collective_timeout_policy::effectiveWorkerJoinTimeoutMs(0)))
    {
        if (!result.completed)
            throw std::runtime_error("Pipeline MTP did not join every participant");
        if (result.exception) std::rethrow_exception(result.exception);
        if (!result.success) return false;
    }
    return true;
}
}

bool PipelineDeviceGeneration::publishOriginalFollowerMTPState(
    size_t index, int physical_rows, std::string *error)
{
    auto &stage = *stages_[index];
    const auto verifier = stage.forward_engine_
        ? stage.forward_engine_->lastAllPositionVerifierForwardGraph() : std::nullopt;
    if (!verifier || !*verifier || !verifier->graph ||
        verifier->signature.seq_len != physical_rows)
    {
        if (error) *error = "Pipeline follower lost its exact grouped verifier";
        return false;
    }
    void *const stream = verifier->stream;
    // Close the verifier's existing transaction borrow after its exact
    // completion event. Followers never admit a second controller.
    if (!stream || !stage.device_generation_state_ready_.handoff.borrowed() ||
        stage.device_generation_state_ready_.handoff.borrowerRole() != DeviceTimelineRole::AllPositionVerifier ||
        !stage.waitForPendingAllPositionVerifierStateReady(stream, "pipeline_initial_mtp_commit") ||
        !stage.materializePipelineFollowerMTPPublicationGraph(*verifier->graph, verifier->signature,
            verifier->signature.seq_len, error) ||
        !stage.executeMTPStatePublicationGraph(stream, error) ||
        !stage.recordAcceptedSpecPublicationReady(stream, "pipeline_initial_mtp_commit") ||
        !stage.publishDeviceGenerationStateReady(stream, 1,
            DeviceGenerationStatePublicationKind::CommittedTransaction))
        return false;
    if (rank_topology_ && !hasLocalTail())
        rank_transaction_terminals_[index].stream = stream;
    return true;
}

bool PipelineDeviceGeneration::publishOriginalRankFollowerState(int physical_rows, std::string *error)
{
    if (!rank_topology_ || hasLocalTail() || phase_ != Phase::Admitted || physical_rows < 2)
    {
        if (error) *error = "Rank follower publication requires its original admitted exact verifier";
        return false;
    }
    phase_ = Phase::Failed;
    std::vector<std::string> failures(stages_.size());
    workers_->dispatch([&, this](size_t index) {
        return publishOriginalFollowerMTPState(index, physical_rows, &failures[index]);
    });
    if (!joinPipelineMTP(*workers_))
    {
        if (error)
            for (size_t index = 0; index < failures.size(); ++index)
                if (!failures[index].empty())
                {
                    *error = "Pipeline participant " + std::to_string(index) + ": " + failures[index];
                    break;
                }
        return false;
    }
    phase_ = Phase::Admitted;
    return true;
}

bool PipelineDeviceGeneration::publishMTPOutcome(
    const DeviceSpeculativePublicationRequest &request, std::string *error)
{
    if (phase_ != Phase::Admitted || !request.valid() || request.requestCount() != 1)
    {
        if (error) *error = "Pipeline publication requires one admitted tail outcome";
        return false;
    }
    phase_ = Phase::Failed;
    std::vector<std::string> failures(stages_.size());
    workers_->dispatch([&, this](size_t index) {
        if (terminalParticipant(index))
            return index != terminalIndex() ||
                tailDomain().runner->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(request, &failures[index]);

        return publishOriginalFollowerMTPState(index, request.physicalVerifierRowsPerRequest(), &failures[index]);
    });
    if (!joinPipelineMTP(*workers_))
    {
        if (error)
            for (size_t index = 0; index < failures.size(); ++index)
                if (!failures[index].empty())
                {
                    *error = "Pipeline participant " + std::to_string(index) + ": " + failures[index];
                    break;
                }
        return false;
    }
    phase_ = Phase::Admitted;
    return true;
}

bool PipelineDeviceGeneration::prepareSpeculativeParticipants(int depth,
    DeviceGenerationLoopTopology topology, DeviceGenerationSamplingMode sampling)
{
    // The terminal participant alone owns request policy. Followers borrow
    // these immutable admission bounds without acquiring a sampler/controller
    // shadow; their device state still arrives over the pipeline command edge.
    const auto &tail_admission = originalAdmission();
    if (!tail_admission)
    {
        LOG_ERROR("Pipeline speculative materialization requires the terminal request admission");
        return false;
    }
    workers_->dispatch([&, this](size_t index) {
        auto &stage = *stages_[index];
        if (terminalParticipant(index))
            return index != terminalIndex() ||
                tailDomain().runner->materializeDeviceResidentGeneration(1, depth, topology, sampling);
        if (transport_ == Transport::NativeDomains && stage.usesParticipantLocalDeviceMoERebalanceController() &&
            sparseQueryExecutionProgramIndex(stage.sparseQueryGraphProgram()) == 0 &&
            !stage.maybeRunDeviceMoERebalanceMaintenanceGraph()) return false;
        const auto verifier = stage.forward_engine_
            ? stage.forward_engine_->lastAllPositionVerifierForwardGraph() : std::nullopt;
        std::string error;
        if (!verifier || !*verifier || !verifier->graph ||
            verifier->signature.device != stage.state_.device_id ||
            !verifier->signature.decode || !verifier->signature.all_position_logits ||
            !verifier->signature.usesDeviceTokenIds() ||
            !verifier->signature.usesDevicePositionIds() ||
            !verifier->signature.usesDeviceSequenceLengths())
            throw std::logic_error("Pipeline follower requires its exact resident grouped verifier");
        auto program_signature = verifier->signature;
        program_signature.sparse_query_execution_program = stage.sparseQueryGraphProgram();
        if (execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)
        {
            auto &loop = stage.deviceGenerationLoopGraph();
            const auto generation = stage.workspaceGeneration(stage.state_.device_id);
            const bool owns_epoch = static_cast<bool>(stage.moe_overlay_epoch_execution_binding_);
            if (stage.mtpVerifierGraphFamily() != MTPVerifierGraphFamily::ExactRows || !loop.stream)
            {
                LOG_ERROR("Pipeline follower lacks its admitted exact verifier family or generation stream"
                          << " participant=" << index << " device=" << stage.state_.device_id.toString());
                return false;
            }
            if (owns_epoch && sparseQueryExecutionProgramIndex(stage.sparseQueryGraphProgram()) == 0 &&
                (!stage.waitForLiveInferenceStateReadyForObservation(loop.stream.get(),
                    "pipeline_exact_mtp_follower_materialization", DeviceTimelineRole::DeviceGenerationController) ||
                 !stage.prepareMoEOverlayEpochForInternalParent(loop.stream.get(),
                    "pipeline_exact_mtp_follower_materialization")))
                return false;
            std::vector<DeviceGraphOrchestrator::MTPPipelineVerifierBranch> candidates;
            const auto widths = stage.mtp_graph_owner_plan_.generationParentWidthCount();
            candidates.reserve(widths);
            for (std::size_t width = 0; width < widths; ++width)
            {
                const int rows = mtpVerifierGraphFamilyWidthAt(width, stage.mtp_max_verifier_rows_,
                    stage.mtpVerifierGraphFamily());
                DeviceGraphOrchestrator::MTPPipelineVerifierBranch branch;
                branch.verifier_signature = stage.retainedMTPVerifierSignature(rows,
                    MTPVerifierOutcomeGraphMode::Disabled, stage.sparseQueryGraphProgram());
                const auto forward = stage.forward_engine_->deviceLoopGraphTemplate(branch.verifier_signature, &error);
                if (!forward || !stage.prepareMTPStatePublicationForVerifier(branch.verifier_signature,
                        loop.stream.get(), &error))
                {
                    LOG_ERROR("Pipeline follower cannot prepare exact verifier/publication rows=" << rows << ": " << error);
                    return false;
                }
                const auto publication = stage.mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(1, rows, &error);
                if (!publication) return false;
                if (owns_epoch)
                {
                    const auto acquire = stage.moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
                        MoEOverlayEpochBoundaryStage::Operation::Acquire, &error);
                    const auto release = stage.moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
                        MoEOverlayEpochBoundaryStage::Operation::Release, &error);
                    if (!acquire || !release) return false;
                    branch.source_fragments.push_back({"pipeline expert epoch acquire", acquire->capture});
                    branch.source_fragments.push_back({"pipeline exact grouped verifier", forward->capture});
                    branch.source_fragments.push_back({"pipeline exact accepted-state publication", publication->capture});
                    branch.source_fragments.push_back({"pipeline expert epoch release", release->capture});
                }
                else
                {
                    branch.source_fragments.push_back({"pipeline exact grouped verifier", forward->capture});
                    branch.source_fragments.push_back({"pipeline exact accepted-state publication", publication->capture});
                }
                candidates.push_back(std::move(branch));
            }
            if (candidates.empty()) return false;
            const bool reuse = loop.valid && loop.workspace_generation == generation &&
                loop.execution_kind == DeviceGraphOrchestrator::MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedPipelineTransaction &&
                loop.pipeline_verifier_branches.size() == candidates.size() &&
                std::equal(loop.pipeline_verifier_branches.begin(), loop.pipeline_verifier_branches.end(), candidates.begin(),
                    [](const auto &left, const auto &right) { return left.hasSameSources(right); });
            if (!reuse)
            {
                loop.invalidateGraph();
                loop.pipeline_verifier_branches = std::move(candidates);
                loop.pipeline_verifier_identity = program_signature;
                loop.workspace_generation = generation;
                loop.request_count = 1;
                loop.draft_depth = depth;
                loop.verifier_rows_per_request = program_signature.seq_len;
                loop.execution_kind = DeviceGraphOrchestrator::MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedPipelineTransaction;
                loop.valid = true;
            }
            if (reuse)
            {
                // Source/signature and workspace authentication above applies
                // to the whole retained family. Count only already-instantiated
                // complete parents, never a prepared fragment list that still
                // needs its first native materialization.
                for (const auto &branch : loop.pipeline_verifier_branches)
                    if (branch.capture && branch.capture->hasExecutable())
                        PerfStatsCollector::addCounter("generation", "pipeline_mtp_follower_graph_reuses", 1.0,
                            "graph_setup", stage.state_.device_id.toString(),
                            {{"execution", deviceGenerationExecutionPolicyName(execution_policy_)},
                             {"physical_rows", std::to_string(branch.verifier_signature.seq_len)}, {"family", "exact_rows"}});
            }
            const auto depths = deviceGenerationSpeculativeDepthRange(tail_admission->depth_policy);
            if (!depths) return false;
            loop.minimum_draft_depth = depths->minimum;
            loop.maximum_draft_depth = depths->maximum;
            return true;
        }
        const auto forward = stage.forward_engine_->deviceLoopGraphTemplate(program_signature, &error);
        // Each export owns its first failure. Publication preparation/export
        // clears its error output and must not hide a rejected forward child.
        if (!forward)
        {
            LOG_ERROR("Pipeline follower cannot export its exact grouped verifier"
                      << " participant=" << index << " device=" << stage.state_.device_id.toString()
                      << " program=" << sparseQueryExecutionProgramIndex(program_signature.sparse_query_execution_program)
                      << " rows=" << program_signature.seq_len << ": " << error);
            return false;
        }
        if (!stage.prepareMTPStatePublicationForVerifier(program_signature,
                stage.deviceGenerationLoopGraph().stream.get(), &error))
        {
            LOG_ERROR("Pipeline follower cannot prepare accepted-state publication"
                      << " participant=" << index << ": " << error);
            return false;
        }
        const auto publication = stage.mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(
            1, verifier->signature.seq_len, &error);
        if (!publication)
        {
            LOG_ERROR("Pipeline follower cannot export accepted-state publication"
                      << " participant=" << index << ": " << error);
            return false;
        }
        auto &loop = stage.deviceGenerationLoopGraph();
        std::vector<DeviceControlledLoopFragment> fragments;
        const bool owns_epoch = static_cast<bool>(stage.moe_overlay_epoch_execution_binding_);
        if (owns_epoch)
        {
            // The first, externally scheduled verifier has published its
            // release. Join the committed state and close any ambient reader
            // before the retained parent's first acquire, just as on the tail.
            if (sparseQueryExecutionProgramIndex(stage.sparseQueryGraphProgram()) == 0 &&
                (!stage.waitForLiveInferenceStateReadyForObservation(loop.stream.get(),
                    "pipeline_mtp_follower_materialization", DeviceTimelineRole::DeviceGenerationController) ||
                !stage.prepareMoEOverlayEpochForInternalParent(loop.stream.get(),
                    "pipeline_mtp_follower_materialization")))
                return false;
            const auto acquire = stage.moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
                MoEOverlayEpochBoundaryStage::Operation::Acquire, &error);
            if (!acquire)
            {
                LOG_ERROR("Pipeline follower lacks its captured epoch acquire: " << error);
                return false;
            }
            const auto release = stage.moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
                MoEOverlayEpochBoundaryStage::Operation::Release, &error);
            if (!release)
            {
                LOG_ERROR("Pipeline follower lacks its captured epoch release: " << error);
                return false;
            }
            fragments.push_back({"pipeline expert epoch acquire", acquire->capture});
            fragments.push_back({"pipeline grouped verifier", forward->capture});
            fragments.push_back({"pipeline accepted-state publication", publication->capture});
            fragments.push_back({"pipeline expert epoch release", release->capture});
        }
        else
        {
            fragments.push_back({"pipeline grouped verifier", forward->capture});
            fragments.push_back({"pipeline accepted-state publication", publication->capture});
        }
        if (const auto *maintenance = nativeStageMaintenance(index))
            fragments.push_back({"original follower cadence and maintenance", maintenance});
        const auto generation = stage.workspaceGeneration(stage.state_.device_id);
        if (loop.valid && loop.workspace_generation == generation &&
            loop.pipeline_verifier_identity == program_signature &&
            std::equal(loop.source_fragments.begin(), loop.source_fragments.end(),
                fragments.begin(), fragments.end(),
                [](const auto &left, const auto &right) { return left.capture == right.capture; }))
        {
            PerfStatsCollector::addCounter("generation", "pipeline_mtp_follower_graph_reuses", 1.0,
                "graph_setup", stage.state_.device_id.toString());
            return true;
        }
        loop.invalidateGraph();
        loop.pipeline_verifier_identity = program_signature;
        loop.source_fragments = std::move(fragments);
        loop.workspace_generation = generation;
        loop.request_count = 1;
        loop.draft_depth = depth;
        loop.verifier_rows_per_request = verifier->signature.seq_len;
        using Kind = DeviceGraphOrchestrator::MTPDeviceGenerationLoopGraphCache::ExecutionKind;
        loop.execution_kind = execution_policy_ == DeviceGenerationExecutionPolicy::NativeConditionalGraph
            ? Kind::NativeConditionalParent : Kind::HostedPipelineTransaction;
        loop.valid = true;
        return true;
    });
    return joinPipelineMTP(*workers_);
}

bool PipelineDeviceGeneration::composeSpeculativeParticipants()
{
    using namespace sampling_math;
    const bool hosted = execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    std::barrier boundary(static_cast<ptrdiff_t>(stages_.size()));
    std::atomic<bool> failed{false};
    workers_->dispatch([&, this](size_t index) {
        auto &stage = *stages_[index];
        auto &loop = stage.deviceGenerationLoopGraph();
        auto &context = GPUDeviceContextPool::instance().getContext(stage.state_.device_id);
        const bool tail = terminalParticipant(index);
        if (hosted)
        {
            if (tail) return true; // Its existing branch table owns every depth.
            (void)stage.retainedStorageTerminalFragments();
            if (loop.pipeline_verifier_branches.size() != stage.mtp_graph_owner_plan_.generationParentWidthCount())
                return false;
            for (auto &branch : loop.pipeline_verifier_branches)
            {
                if (branch.capture && branch.capture->hasExecutable()) continue;
                std::vector<GPUOrderedTimelineStep> steps;
                for (const auto &fragment : branch.source_fragments)
                    steps.push_back({.name = fragment.name, .capture = fragment.capture});
                branch.capture = context.createGraphCapture(loop.stream.get());
                if (!branch.capture || !branch.capture->buildOrderedTimelineTransaction(steps) || !branch.capture->instantiate())
                    return false;
                PerfStatsCollector::addCounter("generation", "pipeline_mtp_follower_graph_materializations", 1.0,
                    "graph_setup", stage.state_.device_id.toString(),
                    {{"execution", deviceGenerationExecutionPolicyName(execution_policy_)},
                     {"physical_rows", std::to_string(branch.verifier_signature.seq_len)}, {"family", "exact_rows"}});
            }
            return true;
        }

        // Record the two terminal command words symmetrically. No participant
        // enters model work or blocks in a conditional predicate during setup.
        loop.capture->reset();
        auto &command = loop.pipeline_children[0];
        std::optional<ScopedBackendGraphCapture> capture;
        bool began = false;
        try
        {
            command = context.createGraphCapture(loop.stream.get());
            if (command) capture.emplace(context, *command, "pipeline_mtp_command");
            began = capture && capture->begin();
        }
        catch (const std::exception &exception)
        {
            LOG_ERROR("Pipeline MTP command capture failed: " << exception.what());
        }
        if (!began) failed = true;
        boundary.arrive_and_wait();
        const bool admitted = !failed.load();
        boundary.arrive_and_wait();
        try
        {
            if (admitted)
            {
                std::vector<LocalTPCollectiveSidebandBuffer> words;
                for (const int field : {kDeviceGenerationControlOk, kDeviceGenerationControlRequestComplete})
                {
                    auto *word = stage.device_generation_storage_.control_device + field;
                    words.push_back({.kind = LocalTPCollectiveSidebandKind::Broadcast,
                        .send_buffer = word, .recv_buffer = word, .element_count = 1,
                        .dtype = CollectiveDataType::INT32,
                        .root_device_index = static_cast<int>(terminalIndex()),
                        .name = "pipeline_mtp_continuation"});
                }
                if (!collective_->collectiveSidebandSpanOnStream(words, static_cast<int>(index),
                        loop.stream.get(), "pipeline_mtp_continuation")) failed = true;
            }
        }
        catch (const std::exception &exception)
        {
            failed = true;
            collective_->requestAbort();
            LOG_ERROR("Pipeline MTP command recording failed: " << exception.what());
        }
        boundary.arrive_and_wait();
        try { if (began) capture->finish(); }
        catch (const std::exception &exception)
        {
            failed = true;
            LOG_ERROR("Pipeline MTP command capture retirement failed: " << exception.what());
        }
        boundary.arrive_and_wait();
        if (failed.load()) return false;

        // Entry overwrites an old request's terminal command before the first
        // predicate. The end-of-iteration exchange follows each local commit;
        // it is the only loop decision on followers, not a mirrored controller.
        const DeviceControlledLoopEntryFragment entry[] = {{"pipeline continuation arrival", command.get()}};
        auto body = loop.source_fragments;
        body.push_back({"pipeline next or terminal command", command.get()});
        const DeviceControlledLoopProgram program{.entry = entry, .iteration = body,.terminal=stage.retainedStorageTerminalFragments()};
        if (tail)
        {
            std::string error;
            if (!DeviceGenerationGraphProgram::native(*loop.capture,
                    {stage.active_device_generation_admission_->depth_policy,
                     stage.device_generation_storage_.control_device,
                     stage.device_generation_storage_.control_stride, 1}, program, error))
            {
                LOG_ERROR("Pipeline MTP tail composition failed: " << error);
                return false;
            }
        }
        else
        {
            const DeviceControlledLoopPredicate predicate{
                .control_rows_device = stage.device_generation_storage_.control_device,
                .control_stride = stage.device_generation_storage_.control_stride, .request_count = 1,
                .healthy_index = kDeviceGenerationControlOk, .complete_index = kDeviceGenerationControlRequestComplete};
            if (!loop.capture->buildDeviceControlledWhileLoop(program, predicate) || !loop.capture->instantiate())
                return false;
            // Observe only the successful complete follower parent. Preparing
            // source fragments is not native graph materialization; reused
            // parents and the tail's separate controller are not counted here.
            PerfStatsCollector::addCounter("generation", "pipeline_mtp_follower_graph_materializations", 1.0,
                "graph_setup", stage.state_.device_id.toString(),
                {{"execution", deviceGenerationExecutionPolicyName(execution_policy_)}});
        }
        return true;
    });
    return joinPipelineMTP(*workers_);
}

bool PipelineDeviceGeneration::launchHostedSpeculativeTransactions()
{
    auto &tail = *stages_[terminalIndex()];
    auto &owner = *tailDomain().runner;
    for (size_t index = 0; index < terminalIndex(); ++index)
    {
        auto &stage = *stages_[index];
        auto &loop = stage.deviceGenerationLoopGraph();
        if (!loop.valid || loop.launched || loop.pipeline_verifier_branches.empty() ||
            std::any_of(loop.pipeline_verifier_branches.begin(), loop.pipeline_verifier_branches.end(),
                [](const auto &branch) { return !branch.capture || !branch.capture->hasExecutable(); }) ||
            loop.workspace_generation != stage.workspaceGeneration(stage.state_.device_id) ||
            !stage.consumeDeviceGenerationStateReady(loop.stream.get(),
                DeviceTimelineRole::DeviceGenerationController, 1, "pipeline_mtp_hosted_admission") ||
            (!tail.hosted_device_generation_cursor_.continuationAdmissionReady() &&
             !stage.prepareMoEOverlayEpochForInternalParent(loop.stream.get(), "pipeline_mtp_hosted_admission")))
            return false;
        loop.launched = true;
    }
    const int bound = tail.active_device_generation_admission_->max_new_tokens;
    for (std::int64_t observation = 0; observation <= bound; ++observation)
    {
        sampling_math::DeviceGenerationDispatchTicket ticket;
        if (!owner.observeDeviceGenerationDispatchTicket(&ticket)) return false;
        // Authenticate the already-observed immutable ticket before selecting
        // any follower executable. The tail's existing submit API performs the
        // state transition only after every follower has been enqueued.
        if (!tail.last_device_generation_dispatch_ticket_ ||
            !ticket.matchesLifecycle(tail.last_device_generation_dispatch_ticket_->sessionEpoch(),
                tail.last_device_generation_dispatch_ticket_->workspaceGeneration()) ||
            !ticket.hasSameDispatchDecision(*tail.last_device_generation_dispatch_ticket_))
            return false;
        if (!advanceHostedFollowerMaintenance()) return false;
        if (rank_topology_ && (!rank_retire_transaction_ || !rank_retire_transaction_(uint64_t(ticket.committed_output_tokens)))) return false;
        if (!ticket.complete)
        {
            if (rank_topology_ && (!rank_publish_transaction_ || !rank_publish_transaction_(
                    MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                    ticket.next_draft_depth + 1, ticket.next_draft_depth))) return false;
            // Validate all participants before the first distributed enqueue.
            for (size_t index = 0; index < terminalIndex(); ++index)
            {
                auto &stage = *stages_[index];
                auto &loop = stage.deviceGenerationLoopGraph();
                const int rows = ticket.next_draft_depth + 1;
                if (ticket.next_draft_depth < loop.minimum_draft_depth ||
                    ticket.next_draft_depth > loop.maximum_draft_depth ||
                    loop.workspace_generation != stage.workspaceGeneration(stage.state_.device_id)) return false;
                const auto branch_index = stage.mtpPublicationWidthIndex(rows);
                if (branch_index >= loop.pipeline_verifier_branches.size()) return false;
                const auto &branch = loop.pipeline_verifier_branches[branch_index];
                if (!branch.capture || !branch.capture->hasExecutable() || branch.verifier_signature !=
                        stage.retainedMTPVerifierSignature(rows, MTPVerifierOutcomeGraphMode::Disabled,
                            stage.sparseQueryGraphProgram())) return false;
            }
            for (size_t index = 0; index < terminalIndex(); ++index)
            {
                auto &stage = *stages_[index];
                auto &loop = stage.deviceGenerationLoopGraph();
                auto &branch = loop.pipeline_verifier_branches[stage.mtpPublicationWidthIndex(ticket.next_draft_depth + 1)];
                if (!submitHostedFollowerCapture(index, *branch.capture)) return false;
                PerfStatsCollector::addCounter("generation", "pipeline_mtp_follower_transaction_submissions", 1.0,
                    "decode", stage.state_.device_id.toString(),
                    {{"epoch_boundary", stage.moe_overlay_epoch_execution_binding_ ? "captured" : "not_required"},
                     {"physical_rows", std::to_string(branch.verifier_signature.seq_len)}, {"family", "exact_rows"}});
            }
        }
        // Submit every follower before the tail can rendezvous in its verifier.
        // The existing tail API authenticates the ticket and selects the branch.
        if (!owner.submitHostScheduledDeviceGenerationAdvance(ticket)) return false;
        if (ticket.complete)
        {
            for (size_t index = 0; index < terminalIndex(); ++index)
                if (!stages_[index]->submitRetainedStorageTerminal()||!stages_[index]->publishDeviceGenerationStateReady(
                        stages_[index]->deviceGenerationLoopGraph().stream.get(), 1,
                        DeviceGenerationStatePublicationKind::Terminal)) return false;
            return true;
        }
    }
    throw std::runtime_error("Pipeline MTP exhausted its response bound without a terminal ticket");
}
}
