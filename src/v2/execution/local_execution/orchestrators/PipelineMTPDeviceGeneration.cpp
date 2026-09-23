/**
 * @file PipelineMTPDeviceGeneration.cpp
 * @brief Compose speculative pipeline generation around one terminal authority.
 *
 * The tail owns drafts, sampling, depth selection and the response budget.
 * Followers own only their layer verifier and accepted-state publication. CUDA
 * receives a continuation command inside a complete native parent; HIP submits
 * one complete follower transaction from the tail's authenticated scheduler
 * ticket. Neither path copies mutable model or controller state to the host.
 */
#include "PipelineDeviceGeneration.h"
#include "DeviceGraphOrchestrator.h"
#include "TPWorkerPool.h"
#include "backends/BackendManager.h"
#include "collective/LocalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"

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
        auto &stage = *stages_[index];
        if (terminalParticipant(index))
            return index != terminalIndex() ||
                domains_.back().runner->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(request, &failures[index]);

        const auto verifier = stage.forward_engine_
            ? stage.forward_engine_->lastAllPositionVerifierForwardGraph() : std::nullopt;
        if (!verifier || !*verifier || !verifier->graph ||
            verifier->signature.seq_len != request.physicalVerifierRowsPerRequest())
        {
            failures[index] = "Pipeline follower lost its exact grouped verifier";
            return false;
        }
        void *const stream = verifier->stream;
        // Close the verifier's existing transaction borrow after its exact
        // completion event. Followers never admit a second controller.
        if (!stream || !stage.device_generation_state_ready_.handoff.borrowed() ||
            stage.device_generation_state_ready_.handoff.borrowerRole() != DeviceTimelineRole::AllPositionVerifier ||
            !stage.waitForPendingAllPositionVerifierStateReady(stream, "pipeline_initial_mtp_commit") ||
            !stage.materializePipelineFollowerMTPPublicationGraph(*verifier->graph,
                verifier->signature.seq_len, &failures[index]) ||
            !stage.executeMTPStatePublicationGraph(stream, &failures[index]) ||
            !stage.recordAcceptedSpecPublicationReady(stream, "pipeline_initial_mtp_commit") ||
            !stage.publishDeviceGenerationStateReady(stream, 1,
                DeviceGenerationStatePublicationKind::CommittedTransaction))
            return false;
        return true;
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
    workers_->dispatch([&, this](size_t index) {
        auto &stage = *stages_[index];
        if (terminalParticipant(index))
            return index != terminalIndex() ||
                domains_.back().runner->materializeDeviceResidentGeneration(1, depth, topology, sampling);
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
        const auto forward = stage.forward_engine_->deviceLoopGraphTemplate(verifier->signature, &error);
        const auto publication = stage.mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(
            1, verifier->signature.seq_len, &error);
        if (!forward || !publication)
        {
            LOG_ERROR("Pipeline follower cannot retain a complete transaction: " << error);
            return false;
        }
        auto &loop = stage.mtp_device_generation_loop_graph_;
        const auto generation = stage.workspaceGeneration(stage.state_.device_id);
        if (loop.valid && loop.workspace_generation == generation &&
            loop.pipeline_verifier_identity == verifier->signature &&
            loop.source_fragments.size() == 2 &&
            loop.source_fragments[0].capture == forward->capture &&
            loop.source_fragments[1].capture == publication->capture)
            return true;
        loop.invalidateGraph();
        loop.pipeline_verifier_identity = verifier->signature;
        loop.source_fragments.push_back({"pipeline grouped verifier", forward->capture});
        loop.source_fragments.push_back({"pipeline accepted-state publication", publication->capture});
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
        auto &loop = stage.mtp_device_generation_loop_graph_;
        auto &context = GPUDeviceContextPool::instance().getContext(stage.state_.device_id);
        const bool tail = terminalParticipant(index);
        if (hosted)
        {
            if (tail) return true; // Its existing branch table owns every depth.
            std::vector<GPUOrderedTimelineStep> steps;
            for (const auto &fragment : loop.source_fragments)
                steps.push_back({.name = fragment.name, .capture = fragment.capture});
            loop.capture->reset();
            return loop.capture->buildOrderedTimelineTransaction(steps) && loop.capture->instantiate();
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
                        .root_device_index = static_cast<int>(stages_.size() - 1),
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
        const DeviceControlledLoopProgram program{.entry = entry, .iteration = body};
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
        }
        return true;
    });
    return joinPipelineMTP(*workers_);
}

bool PipelineDeviceGeneration::launchHostedSpeculativeTransactions()
{
    auto &tail = *stages_[terminalIndex()];
    auto &owner = *domains_.back().runner;
    for (size_t index = 0; index < terminalIndex(); ++index)
    {
        auto &stage = *stages_[index];
        auto &loop = stage.mtp_device_generation_loop_graph_;
        if (!loop.valid || loop.launched || !loop.capture->hasExecutable() ||
            loop.workspace_generation != stage.workspaceGeneration(stage.state_.device_id) ||
            !stage.consumeDeviceGenerationStateReady(loop.stream.get(),
                DeviceTimelineRole::DeviceGenerationController, 1, "pipeline_mtp_hosted_admission"))
            return false;
        loop.launched = true;
    }
    const int bound = tail.active_device_generation_admission_->max_new_tokens;
    for (int observation = 0; observation < bound; ++observation)
    {
        sampling_math::DeviceGenerationDispatchTicket ticket;
        if (!owner.observeDeviceGenerationDispatchTicket(&ticket)) return false;
        if (!ticket.complete)
            for (size_t index = 0; index < terminalIndex(); ++index)
            {
                auto &loop = stages_[index]->mtp_device_generation_loop_graph_;
                if (!loop.capture->launchOnStream(loop.stream.get())) return false;
            }
        // Submit every follower before the tail can rendezvous in its verifier.
        // The existing tail API authenticates the ticket and selects the branch.
        if (!owner.submitHostScheduledDeviceGenerationAdvance(ticket)) return false;
        if (ticket.complete)
        {
            for (size_t index = 0; index < terminalIndex(); ++index)
                if (!stages_[index]->publishDeviceGenerationStateReady(
                        stages_[index]->mtp_device_generation_loop_graph_.stream.get(), 1,
                        DeviceGenerationStatePublicationKind::Terminal)) return false;
            return true;
        }
    }
    throw std::runtime_error("Pipeline MTP exhausted its response bound without a terminal ticket");
}
}
