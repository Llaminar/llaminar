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
 * Hosted ordinary transactions publish each completed tail writer on its exact
 * producer stream before admitting the next ticket-selected writer.
 */
#include "execution/qwen4exp/QwenSparseFollowerCounters.h"
#include "PipelineDeviceGeneration.h"
#include "execution/qwen4exp/QwenSparseQueryProgramObservation.h"
#include "PipelineMaintenanceAdmission.h"
#include "execution/compute_stages/stages/MoEDeviceRebalanceStage.h"
#include "PipelineForwardGraphEdges.h"
#include "Qwen4ExpFixedHeadPipelinePort.h"
#include "execution/qwen4exp/FixedHeadPipelineTerminalEvidence.h"
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "PipelineActivationFanout.h"
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
#include "PipelineNativeOwnership.h"
#include "backends/NativeContextOpaqueContract.h"
#include "execution/global_pp/GlobalPPRankPlan.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include <limits>
#include <barrier>
#include <chrono>
#include <thread>
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

PipelineDeviceGeneration::RankStageEndpoint PipelineDeviceGeneration::inspectOriginalRankStage(
    IInferenceRunner &runner, const RankStageAction &action)
{
    auto *rank = dynamic_cast<IRankOrchestrator *>(&runner);
    const int count = rank ? rank->device_count() : 1;
    const auto *context = count > 1 ? rank->localTPContext() : nullptr;
    if (action.role != RankStageAction::Role::EXECUTE || action.is_global_tp || action.devices.empty() ||
        action.first_layer < 0 || action.last_layer < action.first_layer || action.last_layer == std::numeric_limits<int>::max() ||
        count <= 0 || static_cast<size_t>(count) != action.devices.size() ||
        (count > 1 && (!context || context->degree() != count)))
        throw std::invalid_argument("Rank pipeline projection requires its original executable local stage");
    RankStageEndpoint result;
    std::set<DeviceId> distinct;
    int width = 0;
    for (int member = 0; member < count; ++member)
    {
        auto *stage = dynamic_cast<DeviceGraphOrchestrator *>(rank ? rank->deviceRunner(member) : &runner);
        if (!stage || !stage->pp_stage_config_ || !stage->graph_builder_ ||
            stage->pp_stage_config_->first_layer != action.first_layer ||
            stage->pp_stage_config_->last_layer != action.last_layer + 1 ||
            stage->pp_stage_config_->has_embedding != action.has_embedding ||
            stage->pp_stage_config_->has_lm_head != action.has_lm_head ||
            stage->state_.device_id != action.devices[static_cast<size_t>(member)].toLocalDeviceId() ||
            !stage->state_.device_id.is_gpu() || !distinct.insert(stage->state_.device_id).second)
            throw std::invalid_argument("Rank pipeline projection differs from its original PP/TP stage membership");
        const int participant_width = stage->graph_builder_->config().residualWidth();
        const int participant_verifier_rows = std::max(1,
            resolveMTPRetainedTargetQueryRows(stage->graph_builder_->config().mtp));
        if (!stage->arena_ || !stage->state_.hidden ||
            !stage->physical_memory_authority_ || !stage->physical_memory_authority_->contains(stage->state_.device_id) ||
            stage->state_.activation_seq_len <= 0 || participant_width <= 0 ||
            stage->forward_engine_ || stage->pipeline_forward_edges_ ||
            !stage->deviceGenerationLoopGraph().stream.get())
            throw std::logic_error("Rank pipeline projection requires its original cold arena, PMA and stream");
        // The original arena may not have materialized this bank yet. Cold
        // transport setup allocates it through that arena before binding ports;
        // inspection must neither allocate nor require premature GPU storage.
        if (const auto resident = stage->state_.hidden->current_device();
            resident && *resident != stage->state_.device_id)
            throw std::logic_error("Rank pipeline activation bank has foreign device storage");
        const auto rows = static_cast<size_t>(stage->state_.activation_seq_len);
        if (rows > std::numeric_limits<size_t>::max() / static_cast<size_t>(participant_width) ||
            stage->state_.hidden->numel() < rows * static_cast<size_t>(participant_width))
            throw std::logic_error("Rank pipeline projection exceeds its actual admitted activation bank");
        if (member == 0)
        {
            result = {stage->state_.device_id, stage->state_.activation_seq_len,
                stage->physical_memory_authority_, stage->deviceGenerationLoopGraph().stream.get(), stage->state_.hidden.get(),
                participant_verifier_rows, participant_width};
            width = participant_width;
        }
        else if (stage->physical_memory_authority_ != result.memory ||
            stage->state_.activation_seq_len != result.row_capacity || participant_width != width ||
            participant_verifier_rows != result.verifier_rows)
            throw std::logic_error("Rank pipeline TP siblings have different admitted arena/PMA geometry");
    }
    return result;
}

std::vector<std::unique_ptr<PipelineForwardGraphEdges>> PipelineDeviceGeneration::installOriginalRankStageForwardEdges(
    IInferenceRunner &runner, const RankStageAction &action,
    const PipelineForwardGraphEdges::RankBoundary &boundary)
{
    (void)inspectOriginalRankStage(runner, action);
    if (action.stage_id < 0 || boundary.stage_count < 2 ||
        boundary.stage_index >= boundary.stage_count ||
        boundary.stage_index != static_cast<size_t>(action.stage_id) ||
        action.has_embedding != (boundary.stage_index == 0) ||
        action.has_lm_head != (boundary.stage_index + 1 == boundary.stage_count) ||
        boundary.context || boundary.participant != 0 || boundary.activation_fanout)
        throw std::invalid_argument("Rank pipeline edge installation requires original global roles and neutral TP fields");
    auto *rank = dynamic_cast<IRankOrchestrator *>(&runner);
    const int count = rank ? rank->device_count() : 1;
    auto *context = count > 1 ? rank->localTPContext() : nullptr;
    std::vector<DeviceGraphOrchestrator *> members;
    std::vector<std::unique_ptr<PipelineForwardGraphEdges>> edges;
    members.reserve(static_cast<size_t>(count));
    edges.reserve(static_cast<size_t>(count));
    for (int member = 0; member < count; ++member)
    {
        auto &stage = *dynamic_cast<DeviceGraphOrchestrator *>(rank ? rank->deviceRunner(member) : &runner);
        // Reuse the exact admitted arena bank, just as local captured-domain
        // preparation does. No copied hidden tensor or independent allocator.
        if (!stage.arena_->allocateDeviceStorage(BufferId::HIDDEN_STATE, stage.state_.device_id))
            throw std::logic_error("Rank pipeline activation bank could not materialize its admitted device storage");
        auto local = boundary;
        local.context = context;
        local.participant = member;
        if (action.stage_id > 0 && count > 1)
            local.activation_fanout = PipelineActivationFanout::create(stage.state_.device_id,
                *context, member, boundary.activation_in, stage.physical_memory_authority_,
                stage.deviceGenerationLoopGraph().stream.get());
        edges.push_back(std::make_unique<PipelineForwardGraphEdges>(stage.state_.device_id,
            std::move(local), *stage.arena_, stage.graph_builder_->config().residualWidth()));
        members.push_back(&stage);
    }
    // All throwing validation/construction precedes mutation of graph identity.
    // The returned vector is the sole edge owner, retained by the global parent.
    for (size_t member = 0; member < members.size(); ++member)
        members[member]->pipeline_forward_edges_ = edges[member].get();
    return edges;
}

std::vector<std::unique_ptr<PipelineForwardGraphEdges>> PipelineDeviceGeneration::installOriginalRankStageGenerationEdges(
    IInferenceRunner &runner, const RankStageAction &action,
    const PipelineForwardGraphEdges::CapturedDomain &boundary)
{
    (void)inspectOriginalRankStage(runner, action);
    if (action.stage_id < 0 || boundary.stage_count < 2 ||
        boundary.stage_index >= boundary.stage_count ||
        boundary.stage_index != static_cast<size_t>(action.stage_id) ||
        action.has_embedding != (boundary.stage_index == 0) ||
        action.has_lm_head != (boundary.stage_index + 1 == boundary.stage_count) ||
        boundary.context || boundary.participant != 0 || boundary.activation_fanout)
        throw std::invalid_argument("Rank generation edge installation requires original global roles and neutral TP fields");
    auto *rank = dynamic_cast<IRankOrchestrator *>(&runner);
    const int count = rank ? rank->device_count() : 1;
    auto *context = count > 1 ? rank->localTPContext() : nullptr;
    std::vector<DeviceGraphOrchestrator *> members;
    std::vector<std::unique_ptr<PipelineForwardGraphEdges>> edges;
    members.reserve(static_cast<size_t>(count));
    edges.reserve(static_cast<size_t>(count));
    for (int member = 0; member < count; ++member)
    {
        auto &stage = *dynamic_cast<DeviceGraphOrchestrator *>(rank ? rank->deviceRunner(member) : &runner);
        if (!stage.arena_->allocateDeviceStorage(BufferId::HIDDEN_STATE, stage.state_.device_id))
            throw std::logic_error("Rank generation activation bank could not materialize its admitted device storage");
        std::optional<PipelineForwardGraphEdges::FollowerState> follower;
        if (!action.has_lm_head && stage.graph_builder_->config().mtp.enabled)
        {
            if (stage.mtp_publication_main_kv_base_checkpoints_.empty())
                throw std::logic_error("Rank pipeline verifier lacks its original admitted main-KV checkpoint");
            const auto &checkpoint = stage.mtp_publication_main_kv_base_checkpoints_.front();
            if (!checkpoint.valid() || checkpoint.sequence_index != 0 || checkpoint.device != stage.state_.device_id)
                throw std::logic_error("Rank pipeline verifier checkpoint has foreign or stale ownership");
            follower = PipelineForwardGraphEdges::FollowerState{
                .backend = getBackendFor(stage.state_.device_id),
                .checkpoint = {.cache = stage.state_.kv_cache.get(), .sequence_index = 0,
                    .checkpoint_device = checkpoint.data(), .checkpoint_bytes = checkpoint.bytes}};
        }
        auto local = boundary;
        local.context = context;
        local.participant = member;
        if (action.stage_id > 0 && count > 1)
            local.activation_fanout = PipelineActivationFanout::create(stage.state_.device_id,
                *context, member, boundary.activation_in, stage.physical_memory_authority_,
                stage.deviceGenerationLoopGraph().stream.get());
        edges.push_back(std::make_unique<PipelineForwardGraphEdges>(stage.state_.device_id,
            std::move(local), *stage.arena_, stage.graph_builder_->config().residualWidth(), follower));
        members.push_back(&stage);
    }
    for (size_t member = 0; member < members.size(); ++member)
        members[member]->pipeline_forward_edges_ = edges[member].get();
    return edges;
}

bool PipelineDeviceGeneration::recordOriginalRankProducerCompletion(IBackend &backend, DeviceId device, void *event,
    const RankTransactionTerminal &receipt, uint64_t workspace, uint64_t session, std::string *error)
{
    const auto reject = [&](const char *reason) {
        if (error) *error = reason;
        return false;
    };
    if (error) error->clear();
    if (!device.is_gpu() || backend.backendDeviceType() != device.type || !event)
        return reject("Original rank producer completion lost its exact backend, device or setup-owned event");
    if (!receipt.stream || reinterpret_cast<uintptr_t>(receipt.stream) <= 2 || receipt.recorded)
        return reject("Original rank producer completion has a missing/default stream or already recorded receipt");
    if (!workspace || !session || receipt.workspace != workspace || receipt.session != session)
        return reject("Original rank producer completion has a stale original workspace or request generation");
    switch (receipt.role)
    {
    case MoEOverlayInferenceGraphRole::MainDecode:
    case MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation:
    case MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken:
    case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
        break;
    default:
        return reject("Original rank producer completion has no submitted original condition, decode or verifier body");
    }
    // Receipt creation already selected the real forward/publication/retained
    // parent stream. That producer's ordered native event is the entire ACK
    // obligation; binding live state here would invent a HostResultBridge
    // consumer and steal the next controller's publication ownership.
    if (!backend.recordEvent(event, device.ordinal, receipt.stream))
        return reject("Original rank producer completion could not record its exact native event");
    return true;
}

/** @brief Cold-allocated event set retaining one exact original transaction's native producers. */
struct PipelineDeviceGeneration::RankTransactionCompletion final : IMoEOverlayInferenceCompletionFence
{
    std::vector<DeviceGraphOrchestrator *> stages;
    std::vector<std::shared_ptr<void>> events;
    std::vector<RankTransactionTerminal> identity;
    bool recorded = false;
    std::atomic<MoEOverlayInferenceCompletionFenceProgress> completion{MoEOverlayInferenceCompletionFenceProgress::Pending};

    /** @brief Create persistent native events before any request can submit work. */
    explicit RankTransactionCompletion(const std::vector<DeviceGraphOrchestrator *> &participants)
        : stages(participants), identity(participants.size())
    {
        events.reserve(stages.size());
        for (auto *stage : stages)
        {
            auto *backend = getBackendFor(stage->state_.device_id);
            const auto ordinal = stage->state_.device_id.ordinal;
            void *event = backend ? backend->createEvent(ordinal) : nullptr;
            if (!event) throw std::runtime_error("Original rank completion slot could not allocate its native event");
            events.emplace_back(event, [backend, ordinal](void *value) { backend->destroyEvent(value, ordinal); });
        }
    }

    /** @brief Record every original producer without waiting or changing graph/state ownership. */
    bool record(std::vector<RankTransactionTerminal> &receipts, std::string *error)
    {
        if (receipts.size() != stages.size()) return false;
        if (recorded && completion.load(std::memory_order_acquire) != MoEOverlayInferenceCompletionFenceProgress::Ready)
        {
            if (error) *error = "Original rank native completion slot cannot replace an unproven or failed recording";
            return false;
        }
        // The ring grants reuse only after every old borrower has released its
        // exact slot. Withdraw the old terminal proof before the first new
        // native recording so it can never certify a replacement transaction.
        completion.store(MoEOverlayInferenceCompletionFenceProgress::Pending, std::memory_order_release);
        for (size_t i = 0; i < stages.size(); ++i)
        {
            auto &stage = *stages[i];
            const auto &receipt = receipts[i];
            auto *backend = getBackendFor(stage.state_.device_id);
            std::string receipt_error;
            if (!backend || !PipelineDeviceGeneration::recordOriginalRankProducerCompletion(*backend,
                    stage.state_.device_id, events[i].get(), receipt,
                    stage.workspaceGeneration(stage.state_.device_id), stage.session_epoch_, &receipt_error))
            {
                if (receipt_error.empty()) receipt_error = "Original rank async receipt lost its original backend";
                if (error) *error = receipt_error;
                // Work was already submitted. A partial event record cannot
                // safely return and release graph/workspace leases, nor can an
                // older recording authenticate the missing new producer.
                LOG_ERROR("Original rank async receipt publication failed after native graph submission: " << receipt_error);
                std::terminate();
            }
            identity[i] = receipt;
        }
        recorded = true;
        for (auto &receipt : receipts) receipt = {};
        return true;
    }

    /** @copydoc IMoEOverlayInferenceCompletionFence::poll */
    MoEOverlayInferenceCompletionFenceProgress poll(std::string *error = nullptr) noexcept override
    {
        if (!recorded) return MoEOverlayInferenceCompletionFenceProgress::Failed;
        return pollRetainedInferenceCompletion(completion, [&] {
            bool complete = true;
            for (size_t i = 0; i < stages.size(); ++i)
            {
                auto &stage = *stages[i];
                auto *backend = getBackendFor(stage.state_.device_id);
                bool ready = false;
                if (identity[i].workspace != stage.workspaceGeneration(stage.state_.device_id) ||
                    identity[i].session != stage.session_epoch_ || !backend ||
                    !backend->queryEvent(events[i].get(), stage.state_.device_id.ordinal, &ready))
                {
                    if (error) *error = "Original rank async completion lost its retained workspace/session or native event";
                    return MoEOverlayInferenceCompletionFenceProgress::Failed;
                }
                complete = complete && ready;
            }
            return complete ? MoEOverlayInferenceCompletionFenceProgress::Ready :
                MoEOverlayInferenceCompletionFenceProgress::Pending;
        });
    }
};

void PipelineDeviceGeneration::prepareOriginalRankCompletionSlots(std::size_t slots)
{
    requireLiveParticipantBorrow();
    if (!rank_topology_ || !slots || !rank_completion_slots_.empty() || phase_ != Phase::Idle)
        throw std::logic_error("Original rank completion slots require one cold admitted owner");
    rank_completion_slots_.reserve(slots);
    rank_source_completions_.reserve(slots);
    for (size_t i = 0; i < slots; ++i)
        rank_completion_slots_.push_back(std::make_shared<RankTransactionCompletion>(stages_));
}

std::shared_ptr<IMoEOverlayInferenceCompletionFence>
PipelineDeviceGeneration::recordOriginalRankTransaction(std::string *error)
{
    requireLiveParticipantBorrow();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
        collective_timeout_policy::effectiveCollectTimeoutMs(0));
    while (true)
    {
        // Reclaim only device-proven source leases. Follower leases remain with
        // the service until the real native ACK. No normal submission waits.
        for (auto it = rank_source_completions_.begin(); it != rank_source_completions_.end();)
        {
            const auto state = (*it)->poll(error);
            if (state == MoEOverlayInferenceCompletionFenceProgress::Failed) return {};
            if (state == MoEOverlayInferenceCompletionFenceProgress::Ready) it = rank_source_completions_.erase(it);
            else ++it;
        }
        for (const auto &slot : rank_completion_slots_)
            if (slot.use_count() == 1)
                return slot->record(rank_transaction_terminals_, error) ? slot : nullptr;
        // Waiting is permitted only when reusing a fully occupied retained
        // source ring. A follower cannot invent a slot beyond its source grant.
        if (!hasLocalTail() || rank_source_completions_.empty() ||
            std::chrono::steady_clock::now() >= deadline)
        {
            if (error) *error = "Original rank native completion ring cannot retire its bounded reuse lease";
            return {};
        }
        std::this_thread::yield();
    }
}

bool PipelineDeviceGeneration::retireOriginalRankCompletions(std::string *error)
{
    requireLiveParticipantBorrow();
    // This is the outer terminal ownership boundary, never a per-transaction join.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
        collective_timeout_policy::effectiveCollectTimeoutMs(0));
    while (!rank_source_completions_.empty())
    {
        for (auto it = rank_source_completions_.begin(); it != rank_source_completions_.end();)
        {
            const auto state = (*it)->poll(error);
            if (state == MoEOverlayInferenceCompletionFenceProgress::Failed) return false;
            if (state == MoEOverlayInferenceCompletionFenceProgress::Ready) it = rank_source_completions_.erase(it);
            else ++it;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            if (error) *error = "Original rank command terminal exceeded native completion deadline";
            return false;
        }
        if (!rank_source_completions_.empty()) std::this_thread::yield();
    }
    return true;
}

PipelineDeviceGeneration::PipelineDeviceGeneration(
    const GlobalPPTopology &topology, const GlobalPPRankPlan &rank_plan,
    std::span<const RankStageBinding> stages,
    std::vector<std::unique_ptr<PipelineForwardGraphEdges>> &edges)
    : global_stage_count_(topology.stages.size()), rank_topology_(&topology), rank_plan_(&rank_plan)
{
    const auto actions = rank_plan.executeStages();
    if (topology.numStages() < 2 || !topology.validate().empty() ||
        rank_plan.rank < 0 || rank_plan.rank >= topology.world_size || stages.empty() ||
        actions.size() != stages.size())
        throw std::invalid_argument("Global pipeline cohort requires its complete original local execution plan");
    for (const auto &edge : topology.transfers) (void)edge.physicalConnection();
    std::set<DeviceId> unique;
    size_t next_edge = 0;
    for (size_t index = 0; index < stages.size(); ++index)
    {
        const auto &binding = stages[index];
        if (!binding.runner || !binding.action || binding.action != actions[index])
            throw std::invalid_argument("Global pipeline cohort has a substituted original stage action");
        const auto &action = *binding.action;
        if (action.role != RankStageAction::Role::EXECUTE || action.stage_id < 0 ||
            static_cast<size_t>(action.stage_id) >= topology.stages.size() ||
            action.last_layer == std::numeric_limits<int>::max() ||
            action.has_embedding != (action.stage_id == 0) ||
            action.has_lm_head != (static_cast<size_t>(action.stage_id) + 1 == topology.stages.size()) ||
            (index > 0 && stages[index - 1].action->stage_id >= action.stage_id))
            throw std::invalid_argument("Global pipeline cohort has an invalid original stage coordinate");
        const auto &spec = topology.stages[static_cast<size_t>(action.stage_id)];
        if (spec.is_global_tp || spec.owning_rank != rank_plan.rank ||
            spec.stage_id != action.stage_id || spec.domain_name != action.domain_name ||
            spec.first_layer != action.first_layer || spec.last_layer != action.last_layer ||
            spec.has_embedding != action.has_embedding || spec.has_lm_head != action.has_lm_head ||
            spec.inner_mode != action.inner_mode || spec.devices != action.devices ||
            spec.tp_weights != action.tp_weights || spec.backend != action.backend || action.is_global_tp ||
            (action.inner_mode != InnerParallelism::SINGLE_DEVICE && action.inner_mode != InnerParallelism::LOCAL_TP))
            throw std::invalid_argument("Global pipeline cohort differs from its original rank/stage scope");
        auto *rank = dynamic_cast<IRankOrchestrator *>(binding.runner);
        const int count = rank ? rank->device_count() : 1;
        auto *context = count > 1 ? rank->localTPContext() : nullptr;
        if (count <= 0 || static_cast<size_t>(count) != action.devices.size() ||
            (count > 1 && (!context || context->degree() != count)))
            throw std::invalid_argument("Global pipeline cohort requires exact original local TP membership");
        domains_.push_back({binding.runner, context, stages_.size(), static_cast<size_t>(count),
            static_cast<size_t>(action.stage_id), action.has_embedding, action.has_lm_head});
        for (int member = 0; member < count; ++member)
        {
            auto *stage = dynamic_cast<DeviceGraphOrchestrator *>(rank ? rank->deviceRunner(member) : binding.runner);
            if (!stage || !stage->pp_stage_config_ || !stage->graph_builder_ ||
                stage->pp_stage_config_->first_layer != action.first_layer ||
                stage->pp_stage_config_->last_layer != action.last_layer + 1 ||
                stage->pp_stage_config_->has_embedding != action.has_embedding ||
                stage->pp_stage_config_->has_lm_head != action.has_lm_head ||
                stage->pp_stage_config_->generation_domain != DeviceGenerationExecutionDomain::HostedBoundary ||
                stage->state_.device_id != action.devices[static_cast<size_t>(member)].toLocalDeviceId() ||
                !stage->state_.device_id.is_gpu() || !unique.insert(stage->state_.device_id).second ||
                !stage->state_.hidden || !stage->state_.kv_cache || !stage->arena_ ||
                !stage->physical_memory_authority_ || stage->physical_memory_authority_->worldRank() != rank_plan.rank ||
                !stage->physical_memory_authority_->contains(stage->state_.device_id) ||
                !stage->deviceGenerationLoopGraph().capture || !stage->deviceGenerationLoopGraph().stream ||
                stage->forward_engine_ || next_edge >= edges.size() || !edges[next_edge] ||
                stage->pipeline_forward_edges_ != edges[next_edge].get() ||
                !edges[next_edge]->hasOriginalGenerationBoundary(static_cast<size_t>(action.stage_id),
                    topology.stages.size(), stage->state_.device_id, context, member) ||
                (!stages_.empty() && (stage->physical_memory_authority_ != stages_.front()->physical_memory_authority_ ||
                    stage->graph_builder_->config().residualWidth() != stages_.front()->graph_builder_->config().residualWidth())))
                throw std::invalid_argument("Global pipeline cohort requires original complete captured GPU stage owners");
            // Authenticate borrowed graphs with the same cold factory boundary:
            // a controller-free mutable multi-participant MoE is incomplete,
            // while a true singleton requires no durable movement authority.
            validateMoEDurableResidencyAuthorityForGraph(
                stage->graph_builder_->config(), context, stage->graph_builder_->config().tp_ctx);
            (void)PipelineNativeDomain(stage->state_.device_id, context, member);
            stages_.push_back(stage);
            ++next_edge;
        }
    }
    if (next_edge != edges.size())
        throw std::invalid_argument("Global pipeline cohort has foreign or missing forward-edge owners");
    transport_ = Transport::CapturedDomains;
    execution_policy_ = DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    rank_transaction_terminals_.resize(stages_.size());
    workers_ = std::make_unique<TPWorkerPool>(stages_.size());
    workers_->setFailureCallback([this] { abortParticipants(); });
    // Failed validation/worker creation leaves these original owners with the
    // caller, so installed DGO pointers remain live through failed setup unwind.
    forward_edges_ = std::move(edges);
}

PipelineDeviceGeneration::PipelineDeviceGeneration(
    std::span<const std::unique_ptr<IInferenceRunner>> stages, std::shared_ptr<IMPIContext> original_mpi)
{
    if (stages.size() < 2)
        throw std::invalid_argument("Pipeline generation requires at least two ordered stages");
    global_stage_count_ = stages.size();
    std::vector<GlobalDeviceAddress> devices;
    std::set<DeviceId> unique;
    for (size_t i = 0; i < stages.size(); ++i)
    {
        auto *rank = dynamic_cast<IRankOrchestrator *>(stages[i].get());
        const int count = rank ? rank->device_count() : 1;
        auto *context = count > 1 ? rank->localTPContext() : nullptr;
        if (!stages[i] || count < 1 || (count > 1 && (!context || context->degree() != count)))
            throw std::invalid_argument("Pipeline generation requires exact native TP domain membership");
        domains_.push_back({stages[i].get(), context, stages_.size(), size_t(count), i, i == 0, i + 1 == stages.size()});
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
            const PipelineNativeDomain original_domain(stage->state_.device_id, context, member);
            stages_.push_back(stage);
            devices.push_back(original_domain.memberAddress());
        }
    }
    const auto &fixed_placement = stages_.front()->graph_builder_->config().ngram_table_placement;
    fixed_head_pipeline_ = fixed_placement &&
        fixed_placement->terminalObservationMode() == ImmutableRowTableObservationMode::FixedHeadPipeline;
    const bool one_native_pipeline = std::all_of(stages_.begin(), stages_.end(), [&](const auto *stage) {
        return stage->state_.device_id.type == stages_.front()->state_.device_id.type;
    });
    if (one_native_pipeline)
    {
        if (stages_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("Native pipeline membership exceeds native integer coordinates");
        if (!original_mpi || !stages_.front()->physical_memory_authority_)
            throw std::invalid_argument("Native pipeline has no original admitted MPI/PMA authority");
        std::vector<DeviceId> members;
        std::vector<std::string> prepared_uuids;
        const auto &memory = stages_.front()->physical_memory_authority_;
        for (const auto *stage : stages_) {
            if (stage->physical_memory_authority_ != memory)
                throw std::invalid_argument("Native pipeline stages changed the original shared PMA owner");
            const auto prepared = GPUDeviceContextPool::instance().retainExistingPreparedContext(stage->state_.device_id, memory);
            const auto provenance = GPUDeviceContextPool::instance().retainExistingContextProvenance(stage->state_.device_id);
            prepared.requireWorker(provenance.worker());
            const auto birth = provenance.originalBirthKey();
            if (!birth) throw std::invalid_argument("Native pipeline member has no original prepared source birth");
            members.push_back(stage->state_.device_id);
            prepared_uuids.push_back(birth->physical_uuid);
        }
        requireOriginalNativePipelineOwnership(*original_mpi, memory->worldRank(), members, prepared_uuids);
        pipeline_inventory_ = original_mpi->clusterInventory();
        execution_policy_ = stages_.front()->deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth);
        if (fixed_head_pipeline_ && execution_policy_ == DeviceGenerationExecutionPolicy::NativeConditionalGraph &&
            retainsMTPGraphCapacity(stages_.front()->graph_builder_->config().mtp))
            throw std::invalid_argument(
                "CUDA fixed PLE pipeline cannot prepare retained MTP capacity, including Off with retained MTP graphs. "
                "The native conditional parent requires exact-live verifier data children.");
        collective_ = std::make_unique<LocalTPContext>(std::move(devices), std::vector<float>{},
            stages_.front()->state_.device_id.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        if (fixed_head_pipeline_ || std::any_of(domains_.begin(), domains_.end(), [](const auto &domain) { return domain.count > 1; })) {
            transport_ = Transport::NativeDomains;
            for (std::size_t member = 0; member < terminalIndex(); ++member)
                if (stages_[member]->isMoeRebalancingActive()) {
                    if (!stages_[member]->usesParticipantLocalDeviceMoERebalanceController())
                        throw std::invalid_argument("Native pipeline follower requires its original single-domain cadence service; topology-wide maintenance must use its original service owner");
                    follower_moe_maintenance_ = true;
                    follower_moe_commit_evidence_ |= stages_[member]->graph_builder_->config().mtp.enabled;
                }
        }
    }
    else
    {
        if (fixed_head_pipeline_)
            throw std::invalid_argument("Fixed PLE pipeline requires its original homogeneous native GPU rank");
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
            (transport_ != Transport::NativePipeline &&
             (part.domain_runner != domain.runner || part.domain_context != domain.context ||
              part.domain_participant != i - domain.first || part.edges != stage.pipeline_forward_edges_ ||
              !part.edges || !part.edges->hasDomainBoundary())) || part.device != stage.state_.device_id ||
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
    if (transport_ != Transport::NativePipeline)
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
                // Capture the distinct already-emitted condition topology.
                // Its model state advances without an ordinary cadence publisher;
                // no request, controller or sampler is admitted during setup.
                part.forward = stage.materializeOrdinaryGenerationForward(
                    stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device,
                    OrdinaryForwardRetentionKind::NativeComposable, &error,
                    DeviceGraphOrchestrator::OrdinaryForwardPurpose::ForwardOnlyCondition);
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
                        .dtype = CollectiveDataType::INT32, .root_device_index = static_cast<int>(terminalIndex()),
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
            const bool tail = terminalParticipant(index);
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
    if (transport_ == Transport::NativePipeline || !prefillPrepared()) return false;
    const auto programs = stages_.front()->sparseQueryProgramCount();
    for (auto *stage : stages_)
        if (stage->sparseQueryProgramCount() != programs || !stage->pipeline_forward_edges_ ||
            !stage->pipeline_forward_edges_->hasDomainBoundary()) return false;
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
            // This distinct zero-output source retains the original live-token
            // metadata, activation channels and stage TP collectives. Its typed
            // state transaction excludes the ordinary serial-round publisher.
            part.forward = stage.materializeOrdinaryGenerationForward(
                stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device,
                OrdinaryForwardRetentionKind::NativeComposable, &error,
                DeviceGraphOrchestrator::OrdinaryForwardPurpose::ForwardOnlyCondition);
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
    requireLiveParticipantBorrow();
    // The retained one-shot uses an explicitly zero-output state transaction;
    // it contains no serial cadence publisher or maintenance-clock mutation.
    // Rank TP handles are aggregate ownership tokens, never device pointers.
    // Authenticate through the original terminal domain, then each DGO's own
    // admission consumes its current local mailbox on its exact event edge.
    if (!hasLocalTail() || phase_ != Phase::Idle || request_index != 0 || !forwardOnlyFamilyCurrent() ||
        !logical_state.coversRequest(0)) return false;
    const auto current = tailDomain().runner->deviceResidentLogicalSequenceState();
    if (!logical_state.sameMailboxAs(current)) return false;
    const DeviceGenerationAdmissionRequest admission{
        .request_count = 1, .max_new_tokens = 0,
        .depth_policy = sampling_math::DeviceGenerationPolicy::forwardOnly(),
        .initial_leading_row_disposition = sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted};
    if (!begin(admission)) return false;
    return submitForwardOnly();
}

bool PipelineDeviceGeneration::submitForwardOnly()
{
    requireLiveParticipantBorrow();
    if (phase_ != Phase::Admitted || !forwardOnlyFamilyCurrent()) return false;
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

namespace {
using FixedPLETerminalWords = std::array<std::int32_t, qwen4exp::kFixedHeadPipelineCounterWords>;

/** @brief Project one already completed original generation row without a new device observation. */
void projectFixedPLETerminal(std::span<const int> control, FixedPLETerminalWords &output)
{
    if (control.size() != sampling_math::kDeviceGenerationControlCount)
        throw std::logic_error("Fixed PLE pipeline terminal lost its exact original controller row");
    std::copy_n(control.begin() + qwen4exp::kFixedHeadPipelineFirstWord, output.size(), output.begin());
}

/** @brief Publish only after all original owner events and request result checks have retired. */
void publishFixedPLETerminal(const FixedPLETerminalWords &words, DeviceId device)
{
    const auto totals = qwen4exp::readFixedHeadPipelineCounters(words);
    for (std::size_t field = 0; field < totals.size(); ++field)
        PerfStatsCollector::addCounterWithSequence("immutable_lookup",
            std::string(qwen4exp::kFixedHeadPipelineTerminalMetrics[field]), static_cast<double>(totals[field]),
            {totals[field]}, std::string(qwen4exp::kFixedHeadPipelineTerminalPhase), device.toString(),
            {{"authority", "pipeline_generation_terminal"}, {"request", "0"}});
}
}

bool PipelineDeviceGeneration::finishForwardOnly()
{
    requireLiveParticipantBorrow();
    using namespace sampling_math;
    if (phase_ != Phase::Submitted || !forwardOnlyFamilyCurrent()) return false;
    // Request retirement clears active controller state and can change the
    // selector. Retain the exact executed family before retiring any child;
    // all original completions and observations belong to that same owner.
    auto *const completed_family = forwardOnlyFamily().get();
    phase_ = Phase::Failed;
    auto &tail = *stages_[hasLocalTail() ? terminalIndex() : stages_.size() - 1];
    std::vector<std::array<int, 3>> terminal(stages_.size());
    std::vector<int32_t> published_cached_tokens(stages_.size(), -1);
    std::vector<FixedPLETerminalWords> fixed_ple_terminal(fixed_head_pipeline_ ? stages_.size() : 0);
    // These fixed terminal observations are local host backing. Every accepted
    // copy must retire before they leave scope, including false/throw exits.
    // Existing terminal events own the fence; a failed record/wait is fatal,
    // never permission to destroy storage with a possible native write in flight.
    const auto retire_copies = [&]() noexcept {
        try {
            for (size_t i = 0; i < stages_.size(); ++i)
            {
                const auto &part = completed_family->participants[i];
                if (!part.terminal_copies_pending) continue;
                auto &stage = *stages_[i];
                auto *backend = getBackendFor(part.device);
                if (!backend || !stage.device_generation_terminal_host_ready_event_ ||
                    !backend->recordEvent(stage.device_generation_terminal_host_ready_event_.get(), part.device.ordinal, part.stream))
                    std::terminate();
            }
            for (size_t i = 0; i < stages_.size(); ++i)
            {
                auto &part = completed_family->participants[i];
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
        const auto &part = completed_family->participants[i];
        auto *backend = getBackendFor(part.device);
        if (!stage.consumeDeviceGenerationStateReady(part.stream, DeviceTimelineRole::HostResultBridge,
                1, "pipeline_forward_only_terminal")) return false;
        completed_family->participants[i].terminal_copies_pending = true;
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
        else if (fixed_head_pipeline_ && !backend->deviceToHostOnStream(fixed_ple_terminal[i].data(),
            stage.device_generation_storage_.control_device + qwen4exp::kFixedHeadPipelineFirstWord,
            sizeof(FixedPLETerminalWords), part.device.ordinal, part.stream)) return false;
        // ForwardOnly has no continuation ticket. Consume this already-copied
        // publication window here, before the existing terminal event, so the
        // next Fresh one-shot can preserve only as-yet-unreported prefill work.
        if (fixed_head_pipeline_) {
            auto *control = stage.device_generation_storage_.control_device;
            if (!backend->memset(control + qwen4exp::kFixedHeadPipelineFirstWord, 0,
                    sizeof(FixedPLETerminalWords), part.device.ordinal, part.stream) ||
                !stage.publishPreparedArenaGraphInput(BufferId::MTP_GENERATION_CONTROL, control, part.stream,
                    part.device, "fixed_ple_forward_only_terminal_retire")) return false;
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
        const int expected_complete = transport_ != Transport::NativePipeline && execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions && !terminalParticipant(i) ? 0 : 1;
        if (local[1] != 1 || local[2] != expected_complete || local[0] != terminal[hasLocalTail() ? terminalIndex() : stages_.size() - 1][0]) return false;
        auto &stage = *stages_[i];
        if (terminalParticipant(i)) {
            const auto host_controls = stage.deviceGenerationTerminalHostControls(1);
            if (fixed_head_pipeline_) projectFixedPLETerminal(host_controls, fixed_ple_terminal[i]);
            if (published_cached_tokens[i] != local[0] || !stage.active_device_generation_admission_ ||
                !stage.active_device_generation_admission_->depth_policy.isForwardOnly() ||
                validateDeviceGenerationTerminal(std::span<const int, kDeviceGenerationControlCount>(
                    host_controls.data(), kDeviceGenerationControlCount),
                    *stage.active_device_generation_admission_, stage.device_generation_storage_.response_token_stride,
                    0, 0) != DeviceGenerationTerminalError::None) return false;
        }
        using Metric = qwen4exp::FixedHeadPipelineTerminalMetric;
        if (fixed_head_pipeline_ && qwen4exp::readFixedHeadPipelineCounters(fixed_ple_terminal[i])
            [static_cast<std::size_t>(Metric::Failures)])
            throw std::runtime_error("Fixed PLE forward-only terminal reported immutable range failure on " +
                stage.state_.device_id.toString());
    }
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &part = completed_family->participants[i];
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
        auto &part = completed_family->participants[i];
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
    for (std::size_t i = 0; i < fixed_ple_terminal.size(); ++i)
        publishFixedPLETerminal(fixed_ple_terminal[i], stages_[i]->state_.device_id);
    // The composed native parents bypass direct forward dispatch, including
    // its diagnostic callback boundary. Acquire only their already-written
    // original snapshot manifests after every terminal has completed.
    for (size_t i = 0; i < stages_.size(); ++i) {
        auto &stage = *stages_[i];
        const auto &part = completed_family->participants[i];
        std::string error;
        if (!stage.forward_engine_->publishComposedForwardSnapshots(
                part.forward->signature, part.forward->native()->capture, part.stream, &error)) {
            LOG_ERROR("Pipeline ForwardOnly snapshot publication failed on " << stage.state_.device_id.toString()
                << ": " << error);
            return false;
        }
    }
    // Every original parent has completed and published successfully. Direct
    // executeForward is bypassed by these composed parents; count this logical
    // model forward exactly once on each original child, never at capture time.
    for (size_t i = 0; i < stages_.size(); ++i)
        if (stages_[i]->sparseQueryProgramCount() > 1u)
            recordQwenSparseQueryDecodeForward(stages_[i]->state_.device_id,
                completed_family->participants[i].forward->signature.sparse_query_execution_program,
                ForwardExecutionRole::MainInference, 1);
    PerfStatsCollector::addCounter("generation", "pipeline_forward_only_completed_transactions", 1.0,
        "decode", "rank", {{"sparse_query_program", completed_family->participants[hasLocalTail() ? terminalIndex() : stages_.size() - 1].forward->signature.sparse_query_execution_program ==
            SparseQueryExecutionProgram::ProvenNoRanking ? "proven_no_ranking" : "ranking_possible"},
         {"sampler", "absent"}, {"response_tokens", "0"}, {"model_commits", "1"},
         {"transport", transport_ == Transport::CapturedDomains ? "captured_domains" : "native_pipeline"}});
    terminal_cached_tokens_ = terminal[hasLocalTail() ? terminalIndex() : stages_.size() - 1][0];
    phase_ = Phase::Idle;
    return true;
}

PipelineDeviceGeneration::~PipelineDeviceGeneration()
{
    retireForwardOnlyBeforeStageRelease();
}

void PipelineDeviceGeneration::retireForwardOnlyBeforeStageRelease() noexcept
{
    if (participant_borrow_lifecycle_ == ParticipantBorrowLifecycle::Retired)
        return;
    // Native events outlive every async service return, including command abort.
    // Teardown joins these leases before either graph or workspace owners die.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
        collective_timeout_policy::effectiveCollectTimeoutMs(0));
    for (const auto &slot : rank_completion_slots_)
    {
        if (!slot->recorded) continue;
        while (true)
        {
            std::string completion_error;
            const auto state = slot->poll(&completion_error);
            if (state == MoEOverlayInferenceCompletionFenceProgress::Ready) break;
            if (state == MoEOverlayInferenceCompletionFenceProgress::Failed ||
                std::chrono::steady_clock::now() >= deadline)
            {
                LOG_ERROR("Original rank completion teardown could not retire its recorded producer: "
                    << (completion_error.empty() ? "failed or timed-out native completion proof" : completion_error));
                std::terminate();
            }
            std::this_thread::yield();
        }
    }
    rank_source_completions_.clear();
    rank_completion_slots_.clear();
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
    if (maintenance_admission_ || fixed_head_pipeline_)
    {
        // These families retain actual participant roots across stages.
        // Retire every cloned graph first, while every stage arena and the
        // original flattened communicator still exist. Per-DGO destruction
        // alone would publish the first workspace reusable too early.
        for (auto *stage : stages_)
            stage->retireExecutionTopologyBeforeSharedStorageDetach();
        for (auto *stage : stages_)
        {
            if (stage->pipeline_maintenance_admission_ &&
                stage->pipeline_maintenance_admission_ != maintenance_admission_)
                std::terminate(); // A foreign source may not be retired by this owner.
            stage->pipeline_maintenance_admission_.reset();
            stage->pipeline_maintenance_participant_ = 0;
            stage->pipeline_forward_edges_ = nullptr;
        }
        // The exact edge objects hold the other source references. Every
        // dependent graph is gone; retire ports before source recordings, then
        // scalar roots/controller leases before native context destruction.
        forward_edges_.clear();
        maintenance_admission_.reset();
    }
    // All native completion proofs and graph/source retirements succeeded.
    // Revoke raw participant references now: Rank deliberately retains this
    // transport owner beyond stage destruction, and our destructor re-enters.
    stages_.clear();
    domains_.clear();
    participant_borrow_lifecycle_ = ParticipantBorrowLifecycle::Retired;
}

void PipelineDeviceGeneration::requireLiveParticipantBorrow() const
{
    if (participant_borrow_lifecycle_ != ParticipantBorrowLifecycle::Live)
        throw std::logic_error("Pipeline participant borrow has retired");
}

bool PipelineDeviceGeneration::isGPUStageSet(std::span<const std::unique_ptr<IInferenceRunner>> stages)
{
    if (stages.size() < 2) return false;
    return std::all_of(stages.begin(), stages.end(), [](const auto &stage) {
        return stage && stage->primaryDeviceId().is_gpu();
    });
}

size_t PipelineDeviceGeneration::terminalIndex() const noexcept
{
    for (const auto &domain : domains_)
        if (domain.tail) return domain.first;
    return stages_.size();
}

const PipelineDeviceGeneration::Domain &PipelineDeviceGeneration::tailDomain() const
{
    requireLiveParticipantBorrow();
    for (const auto &domain : domains_)
        if (domain.tail) return domain;
    throw std::logic_error("Pipeline operation requires its original local terminal owner");
}

const std::optional<DeviceGenerationAdmissionRequest> &PipelineDeviceGeneration::originalAdmission() const
{
    requireLiveParticipantBorrow();
    if (rank_topology_) return rank_admission_;
    return stages_[terminalIndex()]->active_device_generation_admission_;
}

const PipelineDeviceGeneration::Domain &PipelineDeviceGeneration::domainFor(size_t participant) const
{
    requireLiveParticipantBorrow();
    for (const auto &domain : domains_)
        if (participant >= domain.first && participant - domain.first < domain.count) return domain;
    throw std::out_of_range("Pipeline participant is outside its frozen domains");
}

void PipelineDeviceGeneration::abortParticipants()
{
    requireLiveParticipantBorrow();
    if (collective_) collective_->requestAbort();
    for (const auto &domain : domains_) if (domain.context) domain.context->requestAbort();
}

bool PipelineDeviceGeneration::supportsMTPPublication() const
{
    requireLiveParticipantBorrow();
    return prefillPrepared() && (!hasLocalTail() || tailDomain().runner->supportsDeviceResidentMTPSpecStatePublication()) &&
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
        const auto geometry = (transport_ == Transport::NativeDomains ? PipelineTransferMemory::nativeForRows :
            PipelineTransferMemory::forRows)(config.residualWidth(), admitted_rows,
                std::max(1, resolveMTPRetainedTargetQueryRows(config.mtp)));
        auto &transfer = TransferEngine::instance();
        const auto create = [&](auto &source, auto &destination, size_t capacity) {
            return transfer.createCapturedTransferChannel(*earlier.physical_memory_authority_, DeviceId::cpu(),
                source.state_.device_id, source.deviceGenerationLoopGraph().stream.get(),
                destination.state_.device_id, destination.deviceGenerationLoopGraph().stream.get(), capacity);
        };
        if (transport_ == Transport::NativeDomains) {
            std::vector<void *> streams;
            for (const auto *stage : stages_) streams.push_back(stage->deviceGenerationLoopGraph().stream.get());
            std::vector<int> activation_consumers, metadata_consumers;
            for (std::size_t member = 0; member < domains_[i].count; ++member)
                activation_consumers.push_back(static_cast<int>(domains_[i].first + member));
            for (std::size_t member = 0; member < domains_[i - 1].count; ++member)
                metadata_consumers.push_back(static_cast<int>(domains_[i - 1].first + member));
            boundaries_.push_back({{}, {},
                PipelineNativeByteChannel::create(*collective_, static_cast<int>(domains_[i - 1].first), activation_consumers,
                    geometry.activation_capacity_bytes, earlier.physical_memory_authority_, streams),
                PipelineNativeByteChannel::create(*collective_, static_cast<int>(domains_[i].first), metadata_consumers,
                    geometry.metadata_capacity_bytes, earlier.physical_memory_authority_, streams)});
        } else {
        boundaries_.push_back({create(earlier, later, geometry.activation_capacity_bytes),
            create(later, earlier, geometry.metadata_capacity_bytes)});
        }
    }
}

bool PipelineDeviceGeneration::prepareServing(const ServingGraphFamilyMaterializationPlan &plan)
{
    requireLiveParticipantBorrow();
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
        std::vector<std::shared_ptr<const Qwen4ExpFixedHeadPipelinePort>> fixed_ports;
        if (fixed_head_pipeline_) {
            const auto &first = *stages_.front();
            const auto &config = first.graph_builder_->config();
            const auto &placement = config.ngram_table_placement;
            if (!collective_ || !pipeline_inventory_ || !config.flash_next || !placement ||
                transport_ != Transport::NativeDomains)
                throw std::logic_error("Fixed PLE native pipeline lost its original admitted graph source");
            const auto &ngram = config.flash_next->ngrams();
            if (ngram.table.dimensions.size() != 2 || ngram.heads != qwen4exp::kNgramHeads ||
                ngram.head_offsets.size() != qwen4exp::kNgramHeads ||
                ngram.head_vocab_sizes.size() != qwen4exp::kNgramHeads ||
                ngram.multipliers.size() != qwen4exp::NgramHashPolicy{}.multipliers.size() ||
                placement->extent() != ImmutableRowTableExtent::fromNative(
                    ngram.table.type, ngram.table.dimensions[0], ngram.row_bytes))
                throw std::invalid_argument("Fixed PLE pipeline changed its original source hash or native geometry");
            qwen4exp::NgramHashPolicy policy;
            std::copy(ngram.multipliers.begin(), ngram.multipliers.end(), policy.multipliers.begin());
            std::copy(ngram.head_offsets.begin(), ngram.head_offsets.end(), policy.offsets.begin());
            std::copy(ngram.head_vocab_sizes.begin(), ngram.head_vocab_sizes.end(), policy.moduli.begin());
            policy.eosToken = ngram.hash_reset_token;
            std::vector<Qwen4ExpFixedHeadPipelinePort::Participant> members;
            for (size_t index = 0; index < stages_.size(); ++index) {
                const auto &stage = *stages_[index];
                const auto &domain = domainFor(index);
                const auto &configured = stage.graph_builder_->config().ngram_table_placement;
                if (!configured || *configured != *placement ||
                    stage.state_.activation_seq_len != placement->fixedHeadPipeline()->maximumRows())
                    throw std::logic_error("Fixed PLE pipeline stage changed its original placement or admitted row envelope");
                members.push_back({stage.state_.device_id, static_cast<int>(domain.global_stage),
                    static_cast<int>(index - domain.first), stage.arena_.get(), stage.injected_model_ctx_.get(),
                    domain.context, stage.deviceGenerationLoopGraph().stream.get()});
            }
            fixed_ports = Qwen4ExpFixedHeadPipelinePort::prepare(placement, members, *collective_,
                pipeline_inventory_->getRank(first.physical_memory_authority_->worldRank()), first.physical_memory_authority_, policy);
        }
        if (follower_moe_maintenance_) {
            // Declare the original stage-local controller family before any
            // model/helper graph freezes pointers. No inference or maintenance
            // executable is launched during this cold preparation wave.
            workers_->dispatch([&, this](std::size_t member) {
                auto &stage = *stages_[member];
                if (!stage.usesParticipantLocalDeviceMoERebalanceController()) return true;
                const auto &domain = domainFor(member);
                if (!domain.context || stage.forward_engine_ || stage.pipeline_forward_edges_)
                    throw std::logic_error("Pipeline maintenance must declare its original workspace before final edges and capture");
                if (!stage.arena_->allocateDeviceStorage(BufferId::HIDDEN_STATE, stage.state_.device_id))
                    throw std::runtime_error("Pipeline maintenance declaration lost its admitted activation bank");
                stage.graph_builder_->prepareForwardGraphTemplate(stage.state_.device_id);
                // Main decode declaration creates the model's original controller
                // binding. Its full serial workspace manifest also fixes the
                // controller bank and generation borrowed by cadence capture.
                // Pipeline wire/derived-budget stages use their already admitted
                // endpoint/arena banks and introduce no WorkspaceAllocator names.
                if (!stage.materializeForwardGraphForShape(1, stage.state_.batch_size,
                        domain.head ? nullptr : stage.state_.hidden.get(),
                        fixed_ports.empty() || !fixed_ports[member]->consumesPLE() ? nullptr : fixed_ports[member]))
                    return false;
                return stage.materializePipelineMaintenanceWithoutLaunch(*domain.context, int(member - domain.first));
            });
            if (!collectPreparation(*workers_)) return false;
        }
        if (follower_moe_commit_evidence_) {
            std::vector<PipelineMaintenanceSource> sources;
            std::vector<void *> streams;
            for (std::size_t member = 0; member < stages_.size(); ++member) {
                auto &stage = *stages_[member];
                streams.push_back(stage.deviceGenerationLoopGraph().stream.get());
                if (!stage.usesParticipantLocalDeviceMoERebalanceController()) { sources.emplace_back(); continue; }
                const MoEDeviceRebalanceStage *owner = nullptr;
                const auto &graph = stage.device_moe_rebalance_maintenance_graph_.graph;
                if (!graph) throw std::logic_error("Pipeline budget has no original stage controller graph");
                for (const auto &name : graph->getExecutionOrder()) {
                    const auto *node = graph->getNode(name);
                    const auto *candidate = node && node->stage ? dynamic_cast<const MoEDeviceRebalanceStage *>(node->stage.get()) : nullptr;
                    if (candidate && candidate->ownsRequestTransactionState()) {
                        if (owner) throw std::logic_error("Pipeline budget found multiple original stage controller owners");
                        owner = candidate;
                    }
                }
                if (!owner) throw std::logic_error("Pipeline budget lost its original controller declaration");
                const auto &declaration = owner->getParams();
                const auto &domain = domainFor(member);
                if (declaration.config.participant_count != domain.count ||
                    declaration.config.participant_id != member - domain.first)
                    throw std::logic_error("Pipeline budget changed original stage/member coordinates");
                auto *workspace = stage.workspace_allocator_->getDeviceWorkspace(stage.state_.device_id);
                const auto name = declaration.workspace_name.empty() ? declaration.stage_name : declaration.workspace_name;
                auto controller = workspace ? workspace->retainBuffer(MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_CONTROLLER_STATE, name), 0, sizeof(DeviceMoERebalanceGraphControllerState)) : nullptr;
                if (!controller || controller->data() != stage.deviceMoERebalanceControllerStateDevice())
                    throw std::logic_error("Pipeline budget substituted its original retained controller workspace");
                sources.push_back(PipelineMaintenanceSource(std::move(controller), declaration.config.initial_maintenance_period_tokens,
                    declaration.config.maintenance_period_tokens));
            }
            maintenance_admission_ = PipelineMaintenanceAdmission::create(*collective_, int(terminalIndex()),
                tailDomain().context, sources, memory, streams);
            for (std::size_t member = 0; member < stages_.size(); ++member) {
                if (stages_[member]->pipeline_maintenance_admission_)
                    throw std::logic_error("Pipeline budget owner cannot replace an original retained family");
                stages_[member]->pipeline_maintenance_admission_ = maintenance_admission_;
                stages_[member]->pipeline_maintenance_participant_ = member;
                if (terminalParticipant(member))
                    stages_[member]->mtp_graph_owner_plan_ = stages_[member]->mtp_graph_owner_plan_.withVerifierPreparationTopology(
                        MTPGraphOwnerPlan::VerifierPreparationTopology::NativePipelineMaintenance);
            }
        }
        // A late edge install would change already captured topology. Reject
        // it before mutating any stage instead of invalidating/recapturing a
        // supposedly ready model. The owner is installed at serving setup.
        for (auto *stage : stages_)
            if (stage->forward_engine_ || stage->pipeline_forward_edges_)
                throw std::logic_error("Pipeline prefill transport must be frozen before forward-engine construction");
        if (transport_ != Transport::NativePipeline)
        {
            // A declared domain edge retains the arena's exact physical bank.
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
            const auto domain_index = domain.global_stage;
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
            std::shared_ptr<PipelineActivationFanout> activation_fanout;
            if (transport_ == Transport::CapturedDomains && domain_index > 0 && domain.count > 1)
                activation_fanout = PipelineActivationFanout::create(stage.state_.device_id,
                    *domain.context, int(index - domain.first), boundaries_[domain_index - 1].activation,
                    stage.physical_memory_authority_, stage.deviceGenerationLoopGraph().stream.get());
            if (transport_ == Transport::NativePipeline)
                forward_edges_.push_back(std::make_unique<PipelineForwardGraphEdges>(*collective_,
                    static_cast<int>(index), *stage.state_.hidden, stage.graph_builder_->config().residualWidth(), follower));
            else
                forward_edges_.push_back(std::make_unique<PipelineForwardGraphEdges>(stage.state_.device_id,
                    PipelineForwardGraphEdges::CapturedDomain{
                        .stage_index = domain_index, .stage_count = global_stage_count_, .context = domain.context,
                        .participant = int(index - domain.first),
                        .activation_in = domain_index ? boundaries_[domain_index - 1].activation : nullptr,
                        .activation_out = domain_index < boundaries_.size() ? boundaries_[domain_index].activation : nullptr,
                        .metadata_in = domain_index < boundaries_.size() ? boundaries_[domain_index].metadata : nullptr,
                        .metadata_out = domain_index ? boundaries_[domain_index - 1].metadata : nullptr,
                        .activation_fanout = std::move(activation_fanout),
                        .native_context = transport_ == Transport::NativeDomains ? collective_.get() : nullptr,
                        .native_participant = transport_ == Transport::NativeDomains ? static_cast<int>(index) : -1,
                        .carries_moe_commit_evidence = follower_moe_commit_evidence_,
                        .maintenance_admission = maintenance_admission_,
                        .native_activation_in = domain_index ? boundaries_[domain_index - 1].native_activation : nullptr,
                        .native_activation_out = domain_index < boundaries_.size() ? boundaries_[domain_index].native_activation : nullptr,
                        .native_metadata_in = domain_index < boundaries_.size() ? boundaries_[domain_index].native_metadata : nullptr,
                        .native_metadata_out = domain_index ? boundaries_[domain_index - 1].native_metadata : nullptr,
                        .fixed_head_port = fixed_ports.empty() ? nullptr : fixed_ports[index]},
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
        local.pipeline_hidden_input = domain.head ? nullptr : stages_[index]->state_.hidden.get();
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
        if (transport_ != Transport::NativePipeline)
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
        PipelineForwardGraphEdges::PublicationBanks banks{metadata.accepted_state_slot_indices, metadata.target_cached_tokens,
            metadata.accepted_state_counts, metadata.publication_ok_flags};
        if (follower_moe_commit_evidence_) {
            auto compact = stage.arena_->getSharedTensor(BufferId::STOCHASTIC_BATCH_OUTPUT_META);
            if (!compact || !compact->gpu_data_ptr() || compact->gpu_data_ptr() != stage.stochastic_batch_output_meta_dev_ ||
                compact->size_bytes() < sampling_math::kSpeculativeBatchMetaCount * sizeof(int32_t))
                throw std::logic_error("Pipeline cadence evidence lost its original compact-field arena bank");
            auto *base = static_cast<int32_t *>(compact->gpu_data_ptr());
            const std::array fields{sampling_math::kSpecBatchMetaOk, sampling_math::kSpecBatchMetaOutputCount,
                sampling_math::kSpecBatchMetaLeadingCommittedOutputCount};
            for (std::size_t field = 0; field < fields.size(); ++field) {
                banks.committed_round_evidence[field] = base + fields[field];
                retained.emplace_back(compact, BufferId::STOCHASTIC_BATCH_OUTPUT_META, fields[field] * sizeof(int32_t), 1);
            }
        }
        stage.pipeline_publication_transport_ = forward_edges_[index]->materializePublicationTransport(
            banks, stage.executor_, *context, gpu, stage.deviceGenerationLoopGraph().stream.get(),
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
    requireLiveParticipantBorrow();
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
    requireLiveParticipantBorrow();
    return forwardMTP(purpose, terminal);
}

bool PipelineDeviceGeneration::forwardDecodeInput(int32_t token_id)
{
    requireLiveParticipantBorrow();
    const int token = token_id;
    return forwardMainInput([&](IInferenceRunner &runner) { return runner.forward(&token, 1); });
}

bool PipelineDeviceGeneration::restoredPrefixMTPDecodeBridge(
    const RestoredPrefixMTPDecodeBridgeRequest &request)
{
    requireLiveParticipantBorrow();
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

bool PipelineDeviceGeneration::forwardOriginalRankFollowerVerifier(int physical_rows)
{
    if (!rank_topology_ || hasLocalTail() || stages_.empty() || physical_rows < 2)
        return false;
    const auto &stage = *stages_.front();
    const auto geometry = resolveMTPServingForwardCaptureGeometry(stage.graph_builder_->config().mtp,
        stage.mtp_max_verifier_rows_, stage.mtpVerifierGraphFamily());
    if (!geometry.enabled || !geometry.valid() || physical_rows > geometry.verifier_rows ||
        !mtpVerifierGraphFamilyContains(physical_rows, geometry.verifier_rows, geometry.family))
        return false;
    return forwardMTP(MTPVerifierForwardPolicy(MTPVerifierOutcomeGraphMode::Disabled,
        physical_rows, geometry), {});
}

bool PipelineDeviceGeneration::forwardMTPVerifier(int logical_rows,
    const std::function<bool(IInferenceRunner &)> &terminal)
{
    requireLiveParticipantBorrow();
    if (!hasLocalTail())
        throw std::logic_error("Logical verifier-width selection belongs to the original terminal stage");
    auto &tail = *stages_[terminalIndex()];
    std::string error;
    const auto width_policy = tail.mtpVerifierPhysicalWidthPolicyForRequest(1, &error);
    const auto geometry = resolveMTPServingForwardCaptureGeometry(tail.graph_builder_->config().mtp,
        tail.mtp_max_verifier_rows_, tail.mtpVerifierGraphFamily());
    const int physical_rows = width_policy ? mtpVerifierPhysicalPaddedSeqLen(
        1, logical_rows, tail.mtp_max_verifier_rows_, *width_policy) : 0;
    if (!tail.graph_builder_->config().mtp.enabled || !geometry.enabled || !geometry.valid() ||
        logical_rows < 2 || physical_rows < logical_rows ||
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
    requireLiveParticipantBorrow();
    const bool original_rank_follower = rank_topology_ && !hasLocalTail() &&
        (phase_ == Phase::Materialized || phase_ == Phase::Submitted);
    if ((!original_rank_follower && phase_ != Phase::Idle && phase_ != Phase::Admitted) || !prefillPrepared() ||
        (hasLocalTail() && !terminal))
        return false;
    const auto previous = phase_;
    phase_ = Phase::Failed;
    workers_->dispatch([&, this](size_t index) {
        auto &stage = *stages_[index];
        if (terminalParticipant(index))
            return index != terminalIndex() || terminal(*tailDomain().runner);
        std::string error;
        void *const stream = stage.explicitGPUStreamForOperation("pipeline_mtp_main_forward");
        // This stream publishes inputs; the retained verifier may execute on
        // another stream. Its live-state prelude must borrow admission on that
        // actual consumer so restored KV/GDN writes precede every graph root.
        const bool ok = stage.executeMTPMainForward(policy,
            domainFor(index).head ? nullptr : stage.state_.hidden.get(),
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
        if (rank_topology_ && !hasLocalTail())
        {
            // Borrow the actual original engine invocation stream now, while
            // its cache owner is alive. A generation stream is not a substitute
            // for a scalar condition/verifier producer on another stream.
            const auto submitted = stage.forward_engine_->lastExecutedForwardGraph();
            if (!submitted || !*submitted || !submitted->stream) return false;
            rank_transaction_terminals_[index].stream = submitted->stream;
        }
        return true;
    });
    if (!collectPreparation(*workers_)) return false;
    phase_ = previous;
    return true;
}

bool PipelineDeviceGeneration::admitOriginalFollower(
    size_t index, const DeviceGenerationAdmissionRequest &request)
{
    auto &stage = *stages_[index];
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
    if (fixed_head_pipeline_ && request.kind == sampling_math::DeviceGenerationAdmissionKind::ContinueResponse &&
        !backend->memset(stage.device_generation_storage_.control_device + qwen4exp::kFixedHeadPipelineFirstWord,
            0, sizeof(FixedPLETerminalWords), stage.state_.device_id.ordinal, stream)) return false;
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
    return true;
}

bool PipelineDeviceGeneration::begin(const DeviceGenerationAdmissionRequest &request)
{
    requireLiveParticipantBorrow();
    if (phase_ != Phase::Idle || !request.valid() || request.request_count != 1)
        return false;
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        const auto &stage = *stages_[i];
        const auto &domain = domainFor(i);
        if (!stage.pp_stage_config_ ||
            stage.pp_stage_config_->has_embedding != domain.head ||
            stage.pp_stage_config_->has_lm_head != terminalParticipant(i) ||
            (collective_ && stage.state_.device_id != collective_->devices()[i].toLocalDeviceId()))
            throw std::logic_error("Pipeline generation stage ownership changed after collective admission");
        (void)PipelineNativeDomain(stage.state_.device_id, domain.context, int(i - domain.first));
        if(!qwen4exp::QwenSparseWorkCounterRow{stage.device_generation_storage_.controlForRequest(0),
                sampling_math::kDeviceGenerationControlCount}.valid())
            throw std::logic_error("Pipeline follower reporting span lost its canonical controller owner/alignment");
    }
    terminal_cached_tokens_.reset();
    if (rank_topology_) rank_admission_ = request;
    // Partial admission must never re-enter an unrelated rank/TP path. Until
    // every participant succeeds, this owner remains in an absorbing failure
    // state; the caller must retire the failed runner, not retry its request.
    phase_ = Phase::Failed;
    // The tail is the only owner permitted to initialize a sampler, response
    // budget or logical next-token authority. Followers admit local work only.
    if (hasLocalTail() && !tailDomain().runner->beginDeviceResidentGeneration(request)) return false;
    for (size_t i = 0; i < terminalIndex(); ++i)
        if (!admitOriginalFollower(i, request)) return false;
    phase_ = Phase::Admitted;
    return true;
}

const IGPUGraphCapture *PipelineDeviceGeneration::nativeStageMaintenance(size_t participant) const
{
    auto &stage = *stages_.at(participant);
    if (execution_policy_ != DeviceGenerationExecutionPolicy::NativeConditionalGraph ||
        !stage.usesParticipantLocalDeviceMoERebalanceController()) return nullptr;
    auto &cache = stage.device_moe_rebalance_maintenance_graph_;
    std::string error;
    if (!cache.graph || cache.workspace_generation != stage.workspaceGeneration(stage.state_.device_id) ||
        !stage.composeDeviceMoERebalanceCadenceTransaction(&error) || !cache.cadence_transaction_capture ||
        !cache.cadence_transaction_capture->hasExecutable())
        throw std::logic_error("Pipeline maintenance lost its complete original CUDA cadence transaction: " + error);
    return cache.cadence_transaction_capture.get();
}

bool PipelineDeviceGeneration::submitHostedFollowerCapture(size_t participant, const IGPUGraphCapture &capture)
{
    auto &stage = *stages_.at(participant);
    auto &loop = stage.deviceGenerationLoopGraph();
    if (terminalParticipant(participant) || !loop.launched || !loop.stream ||
        loop.workspace_generation != stage.workspaceGeneration(stage.state_.device_id) || !capture.hasExecutable()) return false;
    // Materializing a retained child can submit placement maintenance on its
    // separate stream. The child's cold prelude orders that recording stream,
    // not this enclosing parent's replay stream. Join the durable publication
    // here before any ordinary/MTP collective can reuse the same communicator.
    // This is an asynchronous event edge, with no host state observation.
    if (!stage.waitForPendingDeviceMoERebalanceMaintenance(loop.stream.get(),
            DeviceTimelineRole::DeviceGenerationController, "pipeline_follower_captured_transaction"))
        return false;
    if (transport_ != Transport::NativeDomains || !stage.moe_overlay_epoch_execution_binding_)
        return capture.launchOnStream(loop.stream.get());
    auto *backend = getBackendFor(stage.state_.device_id);
    if (!backend || !stage.moe_overlay_epoch_released_event_)
        return false;
    const auto released = stage.moe_overlay_epoch_released_event_.get();
    const int ordinal = stage.state_.device_id.ordinal;
    if (!stage.moe_overlay_epoch_lease_lifecycle_->submitCompleteCapturedHostedParent(
            loop.stream.get(),
            [&](void* stream) { return backend->streamWaitEvent(stream, released, ordinal); },
            [&](void* stream) { return capture.launchOnStream(stream); },
            [&](void* stream) {
                if (!backend->recordEvent(released, ordinal, stream))
                    return false;
                stage.moe_overlay_epoch_acquire_producer_stream_ = nullptr;
                return true;
            }))
        return false;
    PerfStatsCollector::addCounter("moe_overlay_epoch", "pipeline_follower_captured_epoch_terminals", 1.0,
        "decode", stage.state_.device_id.toString(), {{"complete_parent", "true"}, {"blocking", "false"}});
    return true;
}

bool PipelineDeviceGeneration::advanceHostedFollowerMaintenance()
{
    if (transport_ != Transport::NativeDomains) return true;
    for (const auto &domain : domains_) {
        if (domain.tail) continue;
        const auto policy = domain.runner->deviceMoERebalanceMaintenanceExecutionPolicy();
        if (policy == DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive) continue;
        if (policy != DeviceMoERebalanceMaintenanceExecutionPolicy::HostScheduledCapturedMaintenance)
            throw std::logic_error("HIP pipeline follower has no original authenticated maintenance policy");
        // Close each original model/publication stream before any participant's
        // ticket publisher can sample its cadence. This transfers the existing
        // typed handoff; it creates neither an event nor a second state owner.
        for (std::size_t member = domain.first; member < domain.first + domain.count; ++member) {
            auto &stage = *stages_[member];
            const auto &loop = stage.deviceGenerationLoopGraph();
            void *const maintenance = stage.device_moe_rebalance_maintenance_graph_.segment_cache.capture_stream;
            const auto &handoff = stage.device_generation_state_ready_.handoff;
            if (!loop.launched || !maintenance || !handoff.borrowed() ||
                handoff.borrowerRole() != DeviceTimelineRole::DeviceGenerationController ||
                handoff.frontierStream() != loop.stream.get() ||
                !stage.publishDeviceGenerationStateReady(loop.stream.get(), 1,
                    DeviceGenerationStatePublicationKind::CommittedTransaction) ||
                !stage.consumeDeviceGenerationStateReady(maintenance, DeviceTimelineRole::MoERebalanceMaintenance,
                    1, "pipeline_follower_original_maintenance"))
                return false;
        }
        // The original Rank owner compares every actual participant ticket
        // before submitting its complete retained maintenance/ack transaction.
        // A terminal-stage ticket never supplies this stage's maintenance choice.
        DeviceMoERebalanceDispatchTicket ticket;
        if (!domain.runner->observeDeviceMoERebalanceDispatchTicket(&ticket) ||
            !domain.runner->submitHostScheduledDeviceMoERebalanceMaintenance(ticket)) return false;
        for (std::size_t member = domain.first; member < domain.first + domain.count; ++member) {
            auto &stage = *stages_[member];
            const auto &loop = stage.deviceGenerationLoopGraph();
            void *const maintenance = stage.device_moe_rebalance_maintenance_graph_.segment_cache.capture_stream;
            if (!stage.publishDeviceGenerationStateReady(maintenance, 1,
                    DeviceGenerationStatePublicationKind::CommittedTransaction) ||
                !stage.consumeDeviceGenerationStateReady(loop.stream.get(), DeviceTimelineRole::DeviceGenerationController,
                    1, "pipeline_follower_maintenance_return")) return false;
        }
    }
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
            if (transport_ != Transport::NativePipeline)
                return tailDomain().runner->materializeDeviceResidentGeneration(1, 0,
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
        if (transport_ == Transport::NativeDomains && stage.usesParticipantLocalDeviceMoERebalanceController() &&
            sparseQueryExecutionProgramIndex(stage.sparseQueryGraphProgram()) == 0 &&
            !stage.maybeRunDeviceMoERebalanceMaintenanceGraph()) return false;
        const auto *maintenance = nativeStageMaintenance(index);
        const auto forward = stage.materializeOrdinaryGenerationForward(
            stage.device_resident_logical_sequence_state_storage_.next_condition_tokens_device,
            OrdinaryForwardRetentionKind::NativeComposable, &error);
        if (!forward || !forward->native()) { LOG_ERROR("Pipeline follower requires a native-composable forward: " << error); return false; }
        auto &loop = stage.deviceGenerationLoopGraph();
        const auto generation = stage.workspaceGeneration(stage.state_.device_id);
        if (loop.valid && loop.workspace_generation == generation &&
            loop.ordinary_forward_identity == forward->signature && loop.ordinary_maintenance_capture == maintenance && loop.source_fragments.size() == 1 &&
            loop.source_fragments.front().capture == forward->native()->capture)
            return true;
        loop.invalidateGraph();
        loop.ordinary_forward_identity = forward->signature;
        loop.ordinary_maintenance_capture = maintenance;
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
    if (transport_ == Transport::CapturedDomains ||
        (transport_ == Transport::NativeDomains && execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions))
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
        for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
            // Native domain forwards already carry the terminal-authored token.
            if (transport_ == Transport::NativeDomains && field_index == 2) continue;
            auto *field = fields[field_index];
            sidebands.push_back({.kind = LocalTPCollectiveSidebandKind::Broadcast,
                .send_buffer = field, .recv_buffer = field, .element_count = 1,
                .dtype = CollectiveDataType::INT32, .root_device_index = static_cast<int>(terminalIndex()),
                .name = "pipeline_command"});
        }
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
        for (size_t edge = 0; transport_ == Transport::NativePipeline && edge + 1 < stages_.size(); ++edge)
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
        if (transport_ == Transport::NativePipeline && index > 0) body.push_back({"receive pipeline activation", loop.pipeline_children[receive].get()});
        body.push_back(loop.source_fragments.front());
        if (transport_ == Transport::NativePipeline && !tail) body.push_back({"send pipeline activation", loop.pipeline_children[send].get()});
        if (tail) {
            body.push_back({"tail sampler", loop.ordinary_children[OrdinaryGenerationGraphPlan::decode_sampler].get()});
        }
        if (const auto *maintenance = nativeStageMaintenance(index))
            body.push_back({.name = "original stage cadence and maintenance", .capture = maintenance});
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
    requireLiveParticipantBorrow();
    using namespace sampling_math;
    const auto &admission = originalAdmission();
    const auto depths = admission ? deviceGenerationSpeculativeDepthRange(admission->depth_policy) : std::nullopt;
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
        !admission || !admission->valid() || admission->request_count != 1 ||
        admission->kind != DeviceGenerationAdmissionKind::ContinueResponse || !depths ||
        !isValidDeviceGenerationSamplingMode(sampling))
        return reject("requires one admitted speculative ContinueResponse with its exact retained hosted policy");
    // Fixed/Observe sources still schedule their first verifier externally.
    // Their original followers can reuse the already complete pinned verifier
    // parents; only Dynamic source continuation uses native admission ticket zero.
    if (hasLocalTail() && admission->depth_policy.mode != DeviceGenerationPolicyMode::Dynamic)
        return reject("a pinned source response must enter through its externally scheduled first verifier");
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
                    << " expected_depth_range=[" << depths->minimum << ','
                    << depths->maximum << ']' << " retained_widths=" << loop.pipeline_verifier_branches.size());
                return false;
            };
            if (!loop.valid || loop.launched || !loop.stream || loop.request_count != 1 ||
                stage.mtpVerifierGraphFamily() != MTPVerifierGraphFamily::ExactRows ||
                loop.execution_kind != DeviceGraphOrchestrator::MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedPipelineTransaction ||
                loop.workspace_generation != stage.workspaceGeneration(stage.state_.device_id) ||
                loop.minimum_draft_depth != depths->minimum ||
                loop.maximum_draft_depth != depths->maximum ||
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
    if (hasLocalTail() && !tailDomain().runner->prepareDeviceResidentGenerationContinuation(sampling))
        return reject("terminal domain rejected retained-family authentication or native admission-ticket preparation");
    phase_ = Phase::Materialized;
    return true;
}

bool PipelineDeviceGeneration::materialize(int requests, int depth,
    DeviceGenerationLoopTopology topology, DeviceGenerationSamplingMode sampling)
{
    requireLiveParticipantBorrow();
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
        const bool reusable = std::all_of(stages_.begin(), stages_.end(), [this](auto *stage) {
            const auto &loop = stage->deviceGenerationLoopGraph();
            return loop.capture->hasExecutable() &&
                (transport_ != Transport::NativeDomains ||
                 execution_policy_ != DeviceGenerationExecutionPolicy::NativeConditionalGraph || loop.pipeline_children[0]); });
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
    requireLiveParticipantBorrow();
    if (!hasLocalTail() || phase_ != Phase::Materialized) return false;
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
    if (transport_ != Transport::NativePipeline) return launchHostedDomainTransactions();
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
        !tail_loop.ordinary_children[initial]->launchOnStream(tail_loop.stream.get()))
        return false;
    // The pipeline starts the scheduler itself, so ticket observation cannot
    // execute the single-DGO initial-publication block. Close the actual tail
    // writer after its retained child, before a ticket can admit the next one.
    // Followers are already submitted when needed; this only records the
    // existing event on the exact tail stream, without a host wait/readback.
    if (!tail.recordDeviceResidentLogicalSequenceStateMailbox(
            tail.device_generation_storage_.active_request_count,
            tail_loop.stream.get(),
            DeviceGraphOrchestrator::DeviceResidentLogicalStatePublicationKind::OrdinaryGenerationTransaction))
    {
        LOG_ERROR("[PipelineDeviceGeneration] Initial ordinary hosted transaction could not publish logical state");
        return false;
    }
    if (!tail.enqueueDeviceGenerationDispatchTicketObservation())
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
    auto &owner = *tailDomain().runner;
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
            if (!submitHostedFollowerCapture(i, *loop.capture)) return false;
        }
        return true;
    };
    if (tail.deviceGenerationLoopGraph().ordinary_leading ==
            sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted)
    {
        if (rank_topology_ && (!rank_publish_transaction_ ||
                !rank_publish_transaction_(MoEOverlayInferenceGraphRole::MainDecode, 1, 0))) return false;
        if (!submit_followers()) return false;
    }
    const int bound = tail.active_device_generation_admission_->max_new_tokens;
    for (int observation = 0; observation < bound; ++observation)
    {
        sampling_math::DeviceGenerationDispatchTicket ticket;
        if (!owner.observeDeviceGenerationDispatchTicket(&ticket) || !advanceHostedFollowerMaintenance()) return false;
        // This immutable ticket authenticates completion of the preceding tail
        // body. Only now may source protocol slots release remote return owners.
        if (rank_topology_ && (!rank_retire_transaction_ || !rank_retire_transaction_(uint64_t(ticket.committed_output_tokens)))) return false;
        if (!ticket.complete)
        {
            if (rank_topology_ && (!rank_publish_transaction_ ||
                    !rank_publish_transaction_(MoEOverlayInferenceGraphRole::MainDecode, 1, 0))) return false;
            if (!submit_followers()) return false;
        }
        if (!owner.submitHostScheduledDeviceGenerationAdvance(ticket)) return false;
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
    requireLiveParticipantBorrow();
    if (!hasLocalTail() || phase_ != Phase::Submitted || !result) return false;
    phase_ = Phase::Failed;
    *result = {};
    DeviceGenerationTerminalResult terminal_result;
    if (!tailDomain().runner->finishDeviceResidentGeneration(&terminal_result)) return false;
    struct TerminalObservation {
        std::array<int,3> state{}; ///< Existing independent KV/health terminal evidence.
        QwenSparseFollowerCounterObservation counters; ///< Exact final32B, never execution state.
        FixedPLETerminalWords fixed_ple{};
        int fixed_ple_error = 0;
        bool copies_pending = false;
    };
    std::vector<TerminalObservation> terminal(stages_.size());
    const auto retire_copies = [&]() noexcept {
        for (size_t i = 0; i < stages_.size(); ++i) {
            if (!terminal[i].copies_pending) continue;
            auto &stage = *stages_[i];
            auto *backend = getBackendFor(stage.state_.device_id);
            void *stream = stage.deviceGenerationLoopGraph().stream.get();
            if (!backend || !backend->recordEvent(stage.device_generation_terminal_host_ready_event_.get(),
                    stage.state_.device_id.ordinal, stream) ||
                !backend->waitForEvent(stage.device_generation_terminal_host_ready_event_.get(), stage.state_.device_id.ordinal))
                std::terminate();
        }
    };
    struct CopyRetirement {
        const decltype(retire_copies) &retire;
        /** @brief Keep every accepted terminal copy destination until the original event on exceptional exit. */
        ~CopyRetirement() noexcept { retire(); }
    } copy_retirement{retire_copies};
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto *backend = getBackendFor(stage.state_.device_id);
        void *stream = stage.deviceGenerationLoopGraph().stream.get();
        const bool follower = !terminalParticipant(i);
        if (follower && !stage.consumeDeviceGenerationStateReady(stream, DeviceTimelineRole::HostResultBridge,
                1, "pipeline_follower_terminal")) return false;
        terminal[i].copies_pending = true;
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
        if (fixed_head_pipeline_) {
            if (follower) {
                if (!backend->deviceToHostOnStream(terminal[i].fixed_ple.data(),
                        stage.device_generation_storage_.control_device + qwen4exp::kFixedHeadPipelineFirstWord,
                        sizeof(FixedPLETerminalWords), stage.state_.device_id.ordinal, stream) ||
                    !backend->deviceToHostOnStream(&terminal[i].fixed_ple_error,
                        stage.device_generation_storage_.control_device + sampling_math::kDeviceGenerationControlErrorCode,
                        sizeof(int), stage.state_.device_id.ordinal, stream)) return false;
            } else {
                projectFixedPLETerminal(stage.deviceGenerationTerminalHostControls(1), terminal[i].fixed_ple);
            }
        }
        if (!backend->recordEvent(stage.device_generation_terminal_host_ready_event_.get(),
                stage.state_.device_id.gpu_ordinal(), stream)) return false;
    }
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        const bool hosted_follower = !terminalParticipant(i) &&
            execution_policy_ == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
        if (!getBackendFor(stage.state_.device_id)->waitForEvent(stage.device_generation_terminal_host_ready_event_.get(),
                stage.state_.device_id.gpu_ordinal())) return false;
        terminal[i].copies_pending = false;
        if (!hosted_follower && (terminal[i].state[1] != 1 || terminal[i].state[2] != 1)) return false;
        if (!terminalParticipant(i))
        {
            stage.awaitRetainedStorageTerminal();
            if (!stage.deviceGenerationLoopGraph().retireCompletedLaunch() ||
                !stage.device_generation_state_ready_.handoff.retireAfterHostCompletion(1)) return false;
            stage.device_generation_storage_.active_request_count = 0;
        }
    }
    for (std::size_t i = 0; i < terminal.size(); ++i) {
        const auto &local = terminal[i];
        if (local.state[0] != terminal.back().state[0])
            throw std::runtime_error("Pipeline generation terminal KV positions disagree; no state repair is permitted");
        using Metric = qwen4exp::FixedHeadPipelineTerminalMetric;
        if (fixed_head_pipeline_ && (local.fixed_ple_error ||
            qwen4exp::readFixedHeadPipelineCounters(local.fixed_ple)[static_cast<std::size_t>(Metric::Failures)])) {
            throw std::runtime_error("Fixed PLE pipeline reported original immutable range failure on " +
                stages_[i]->state_.device_id.toString());
        }
    }
    // The tail reported its own completed forwards through the ordinary DGO
    // terminal. Followers report only after the independent KV check above;
    // they borrow its authenticated count, never a mirrored response ledger.
    for (size_t i = 0; i < terminalIndex(); ++i)
        if (stages_[i]->deviceGenerationLoopGraph().draft_depth == 0 &&
            !stages_[i]->observeOrdinaryGenerationForwardCompletion(
                terminal_result.requests.front().published_state_commit_count)) return false;
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
    if (fixed_head_pipeline_)
        for (std::size_t i = 0; i < terminal.size(); ++i)
            publishFixedPLETerminal(terminal[i].fixed_ple, stages_[i]->state_.device_id);
    PerfStatsCollector::addCounter("generation", "pipeline_terminal_responses", 1.0, "decode", "rank",
        {{"response_owner", "tail"}, {"validation", "independent_stage_kv_positions"}});
    terminal_cached_tokens_ = terminal.front().state[0];
    *result = std::move(terminal_result);
    rank_admission_.reset();
    phase_ = Phase::Idle;
    return true;
}

bool PipelineDeviceGeneration::beginOriginalRankFollowerCommand(
    const DeviceGenerationAdmissionRequest &request,
    DeviceGenerationSamplingMode sampling, std::string *error)
{
    requireLiveParticipantBorrow();
    if (error) error->clear();
    if (!rank_topology_ || hasLocalTail() || !isValidDeviceGenerationSamplingMode(sampling) ||
        !begin(request))
    {
        if (error) *error = "Original rank follower command has invalid ownership or admission";
        return false;
    }
    rank_sampling_ = sampling;
    rank_initial_verifier_complete_ = false;
    if (request.depth_policy.isForwardOnly()) return forwardOnlyFamilyCurrent();
    // Preserve the proven speculative family across requests. An ordinary
    // source is unrelated to that topology and must not overwrite it merely
    // to admit a new speculative request.
    if (!request.depth_policy.isOrdinary())
    {
        if (request.kind == sampling_math::DeviceGenerationAdmissionKind::ContinueResponse)
        {
            if (!prepareContinuation(sampling)) return false;
            rank_initial_verifier_complete_ = true;
        }
        return true;
    }
    if (!prepareParticipants(sampling) || !composeParticipants())
    {
        phase_ = Phase::Failed;
        if (error) *error = "Original rank follower could not retain its complete ordinary body";
        return false;
    }
    phase_ = Phase::Materialized;
    return true;
}

bool PipelineDeviceGeneration::joinOriginalRankTransaction(std::string *error)
{
    requireLiveParticipantBorrow();
    const auto reject = [&](const char *message) {
        phase_ = Phase::Failed;
        if (error) *error = message;
        return false;
    };
    if (!rank_topology_ || rank_transaction_terminals_.size() != stages_.size())
        return reject("Original rank transaction has no complete local completion receipts");
    // Submit all event publications before querying any participant. Every
    // original parent already owns its precise submission stream; semantic
    // forwarding joins are explicit event DAG edges on that stream.
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &receipt = rank_transaction_terminals_[i];
        auto *backend = getBackendFor(stage.state_.device_id);
        if (!backend || !receipt.stream || reinterpret_cast<uintptr_t>(receipt.stream) <= 2 ||
            receipt.role == MoEOverlayInferenceGraphRole::None || receipt.recorded ||
            receipt.workspace != stage.workspaceGeneration(stage.state_.device_id) ||
            receipt.session != stage.session_epoch_ || !stage.device_generation_terminal_host_ready_event_ ||
            !stage.waitForLiveInferenceStateReadyForObservation(receipt.stream,
                "original_rank_transaction_terminal", DeviceTimelineRole::HostResultBridge) ||
            !backend->recordEvent(stage.device_generation_terminal_host_ready_event_.get(),
                stage.state_.device_id.ordinal, receipt.stream))
            return reject("Original rank transaction could not publish its exact native completion receipt");
        receipt.recorded = true;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
        collective_timeout_policy::effectiveCollectTimeoutMs(0));
    while (true)
    {
        bool complete = true;
        for (size_t i = 0; i < stages_.size(); ++i)
        {
            const auto &stage = *stages_[i];
            const auto &receipt = rank_transaction_terminals_[i];
            bool ready = false;
            if (!receipt.recorded || receipt.workspace != stage.workspaceGeneration(stage.state_.device_id) ||
                receipt.session != stage.session_epoch_ ||
                !getBackendFor(stage.state_.device_id)->queryEvent(
                    stage.device_generation_terminal_host_ready_event_.get(), stage.state_.device_id.ordinal, &ready))
                return reject("Original rank transaction native completion identity/query failed");
            complete = complete && ready;
        }
        if (complete)
        {
            for (auto &receipt : rank_transaction_terminals_) receipt = {};
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline)
            return reject("Original rank transaction native completion exceeded its collective deadline");
        std::this_thread::yield();
    }
}

bool PipelineDeviceGeneration::submitOriginalRankFollowerBody(int physical_rows, std::string *error)
{
    requireLiveParticipantBorrow();
    if (!rank_topology_ || hasLocalTail() || !rank_sampling_ ||
        (phase_ != Phase::Materialized && phase_ != Phase::Submitted)) return false;
    // Authenticate every local branch before the first distributed enqueue.
    for (auto *stage : stages_)
    {
        const auto &loop = stage->deviceGenerationLoopGraph();
        if (!loop.valid || !loop.stream ||
            loop.workspace_generation != stage->workspaceGeneration(stage->state_.device_id)) return false;
        if (physical_rows == 1)
        {
            if (!loop.capture || !loop.capture->hasExecutable()) return false;
        }
        else
        {
            const auto width = stage->mtpPublicationWidthIndex(physical_rows);
            if (width >= loop.pipeline_verifier_branches.size()) return false;
            const auto &branch = loop.pipeline_verifier_branches[width];
            if (!branch.capture || !branch.capture->hasExecutable() || branch.verifier_signature !=
                stage->retainedMTPVerifierSignature(physical_rows, MTPVerifierOutcomeGraphMode::Disabled,
                    stage->sparseQueryGraphProgram())) return false;
        }
    }
    for (auto *stage : stages_)
    {
        auto &loop = stage->deviceGenerationLoopGraph();
        if (!loop.launched)
        {
            // The actual retained controller, not its ACK fence, inherits the
            // externally scheduled first verifier's accepted-state publication.
            if (!stage->joinPublishedLiveStateHandoffs(loop.stream.get(),
                    DeviceTimelineRole::DeviceGenerationController, "original_rank_follower_submission") ||
                !stage->consumeDeviceGenerationStateReady(loop.stream.get(),
                    DeviceTimelineRole::DeviceGenerationController, 1, "original_rank_follower_submission")) return false;
            loop.launched = true;
        }
        auto *capture = physical_rows == 1 ? loop.capture.get() :
            loop.pipeline_verifier_branches[stage->mtpPublicationWidthIndex(physical_rows)].capture.get();
        if (!capture->launchOnStream(loop.stream.get()))
        {
            phase_ = Phase::Failed;
            if (error) *error = "Original rank follower failed after partial retained-parent submission";
            return false;
        }
        // This counter proves a real retained exact-width verifier parent was
        // submitted on this original participant's authenticated graph stream.
        if (physical_rows > 1)
            PerfStatsCollector::addCounter("generation", "pipeline_mtp_follower_transaction_submissions", 1.0,
                "decode", stage->state_.device_id.toString(),
                {{"epoch_boundary", stage->moe_overlay_epoch_execution_binding_ ? "captured" : "not_required"},
                 {"physical_rows", std::to_string(physical_rows)}, {"family", "exact_rows"},
                 {"world_rank", std::to_string(rank_plan_->rank)}, {"authority", "original_rank"}});
    }
    phase_ = Phase::Submitted;
    return true;
}

bool PipelineDeviceGeneration::executeMoEOverlayInferenceTransaction(
    const MoEOverlayInferenceTransactionTicket &, std::string *error)
{
    requireLiveParticipantBorrow();
    if (error) *error = "Original GPU rank transactions require typed asynchronous submission";
    return false;
}

MoEOverlayInferenceAsyncSubmission PipelineDeviceGeneration::submitMoEOverlayInferenceTransaction(
    const MoEOverlayInferenceTransactionTicket &ticket, std::string *error)
{
    requireLiveParticipantBorrow();
    using Progress = MoEOverlayInferenceCompletionFenceProgress;
    if (!enqueueOriginalRankTransaction(ticket, error)) return {.progress = Progress::Failed};
    if (ticket.graph_role == MoEOverlayInferenceGraphRole::GenerationAdmission ||
        ticket.graph_role == MoEOverlayInferenceGraphRole::OrdinaryForwardOnly)
        return {.progress = Progress::Ready};
    auto completion = recordOriginalRankTransaction(error);
    if (!completion) return {.progress = Progress::Failed};
    return {.progress = Progress::Pending, .completion = std::move(completion)};
}

bool PipelineDeviceGeneration::enqueueOriginalRankTransaction(
    const MoEOverlayInferenceTransactionTicket &ticket, std::string *error)
{
    requireLiveParticipantBorrow();
    if (error) error->clear();
    if (!rank_topology_ || hasLocalTail() ||
        ticket.action != MoEOverlayInferenceTransactionAction::Execute ||
        ticket.source_world_rank != rank_topology_->stages.back().owning_rank ||
        ticket.target_world_rank != rank_plan_->rank || ticket.request_count != 1 ||
        ticket.logical_rows_per_request <= 0 ||
        ticket.physical_rows_per_request < ticket.logical_rows_per_request)
    {
        if (error) *error = "Original rank executor rejected source, target, command, or geometry";
        return false;
    }
    if (ticket.graph_role == MoEOverlayInferenceGraphRole::GenerationAdmission)
        return ticket.logical_rows_per_request == 1 && ticket.physical_rows_per_request == 1 &&
            !rank_admission_ && rank_receive_admission_ && rank_receive_admission_(error);
    const bool scalar_condition =
        ticket.graph_role == MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation ||
        ticket.graph_role == MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken;
    if (!scalar_condition && (!rank_admission_ || !rank_sampling_)) return false;
    if (ticket.graph_role == MoEOverlayInferenceGraphRole::OrdinaryForwardOnly)
    {
        if (!rank_admission_->depth_policy.isForwardOnly() || ticket.logical_rows_per_request != 1 ||
            ticket.physical_rows_per_request != 1 || !submitForwardOnly())
        {
            if (error) *error = "Original ForwardOnly ticket did not retire its exact sampler-free parent";
            return false;
        }
        return true;
    }
    // Protocol already authenticated the source's placement epoch and immutable
    // topology/workspace namespace. It is intentionally never compared with a
    // different rank's local placement or workspace generation here.
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &receipt = rank_transaction_terminals_[i];
        if (receipt.recorded || receipt.stream) return false;
        receipt.stream = stage.deviceGenerationLoopGraph().stream.get();
        receipt.workspace = stage.workspaceGeneration(stage.state_.device_id);
        receipt.session = stage.session_epoch_;
        receipt.role = ticket.graph_role;
    }
    bool submitted = false;
    switch (ticket.graph_role)
    {
    case MoEOverlayInferenceGraphRole::MainDecode:
        submitted = ticket.logical_rows_per_request == 1 && ticket.physical_rows_per_request == 1 &&
            submitOriginalRankFollowerBody(1, error);
        break;
    case MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation:
    case MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken:
        submitted = ticket.logical_rows_per_request == 1 && ticket.physical_rows_per_request == 1 &&
            forwardMTPCondition(moeOverlayPipelineConditionForwardPurpose(ticket.graph_role), {});
        if (submitted)
        {
            // Scalar conditions use the retained forward family's exact stream,
            // including the initial condition before a generation parent exists.
            // A loop stream cannot authenticate completion of that submission.
            for (size_t i = 0; i < stages_.size(); ++i)
            {
                const auto forward = stages_[i]->forward_engine_ ?
                    stages_[i]->forward_engine_->lastExecutedForwardGraph() : std::nullopt;
                if (!forward || !*forward || !forward->stream)
                {
                    submitted = false;
                    break;
                }
                rank_transaction_terminals_[i].stream = forward->stream;
            }
        }
        break;
    case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
        if (ticket.draft_depth <= 0 || ticket.draft_depth + 1 != ticket.logical_rows_per_request)
            break;
        if (!rank_initial_verifier_complete_)
        {
            submitted = forwardOriginalRankFollowerVerifier(ticket.physical_rows_per_request);
            if (submitted)
            {
                phase_ = Phase::Admitted;
                submitted = publishOriginalRankFollowerState(ticket.physical_rows_per_request, error);
            }
            // Publication records its real producer. The retained branch family
            // borrows stable cache owners and consumes their explicit ready
            // event; materialization does not require a host completion join.
            if (submitted) submitted = materialize(1, ticket.draft_depth,
                rank_admission_->depth_policy.mode == sampling_math::DeviceGenerationPolicyMode::Dynamic ?
                    DeviceGenerationLoopTopology::DynamicDepth : DeviceGenerationLoopTopology::FixedDepth,
                *rank_sampling_);
            if (submitted) rank_initial_verifier_complete_ = true;
        }
        else submitted = submitOriginalRankFollowerBody(ticket.physical_rows_per_request, error);
        break;
    default:
        break; // A rank follower owns neither a vocabulary sidecar nor a sampler.
    }
    if (!submitted)
    {
        phase_ = Phase::Failed;
        abortParticipants();
        if (error && error->empty()) *error = "Original rank follower rejected its exact retained graph transaction";
        return false;
    }
    return true;
}


bool PipelineDeviceGeneration::finishOriginalRankFollowerCommand(
    int source_terminal_commit_count, int expected_source_cached_tokens, std::string *error)
{
    requireLiveParticipantBorrow();
    if (error) error->clear();
    if (rank_topology_ && !hasLocalTail() && rank_admission_ &&
        rank_admission_->depth_policy.isForwardOnly() && phase_ == Phase::Idle)
    {
        if (source_terminal_commit_count != 1 || !terminal_cached_tokens_ ||
            *terminal_cached_tokens_ != expected_source_cached_tokens) return false;
        rank_admission_.reset();
        rank_sampling_.reset();
        return true;
    }
    if (!rank_topology_ || hasLocalTail() || !rank_admission_ || !rank_sampling_ ||
        (phase_ != Phase::Submitted && phase_ != Phase::Materialized && phase_ != Phase::Admitted) ||
        source_terminal_commit_count < 0 || expected_source_cached_tokens < 0)
    {
        if (error) *error = "Original rank follower terminal has incomplete original admission or source observation";
        return false;
    }
    phase_ = Phase::Failed;
    struct Observation
    {
        int cached_tokens = -1; ///< Validation-only original local KV terminal.
        QwenSparseFollowerCounterObservation counters; ///< Existing compact final counters.
    };
    std::vector<Observation> observations(stages_.size());
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &loop = stage.deviceGenerationLoopGraph();
        auto *backend = getBackendFor(stage.state_.device_id);
        void *stream = loop.stream.get();
        auto &receipt = rank_transaction_terminals_[i];
        const auto fail_terminal = [&](const char *reason) -> bool {
            if (i > 0)
            {
                LOG_ERROR(reason << "; prior original follower terminal host copies remain outstanding");
                std::terminate();
            }
            if (error) *error = reason;
            return false;
        };
        if (receipt.recorded || receipt.stream)
            return fail_terminal("Original rank follower terminal retained an unretired transaction receipt");
        if (!backend || !stream)
            return fail_terminal("Original rank follower terminal lost its backend or exact retained graph stream");
        auto &handoff = stage.device_generation_state_ready_.handoff;
        // The tail can finish a one-token response using the prefill sample
        // without submitting any follower forward. Its admission publication
        // still owns the reset/counter writes: acquire that original event on
        // the terminal stream before closing storage or publishing Terminal.
        // A first MTP verifier can likewise leave a committed publication. A
        // prior retained body already owns this stream and must not borrow twice.
        if (handoff.published())
        {
            if (handoff.lastPublication() == DeviceGenerationStatePublicationKind::Terminal)
                return fail_terminal("Original rank follower terminal attempted to replace an existing terminal publication");
            // A response can end after its first externally scheduled verifier.
            // Its terminal controller still inherits that publication through
            // the declared producer->controller edge before final host export.
            if (!stage.joinPublishedLiveStateHandoffs(stream, DeviceTimelineRole::DeviceGenerationController,
                    "original_rank_follower_terminal_acquire"))
                return fail_terminal("Original rank follower terminal could not inherit its published live-state producer");
            if (!stage.consumeDeviceGenerationStateReady(stream, DeviceTimelineRole::DeviceGenerationController,
                    1, "original_rank_follower_terminal_acquire"))
                return fail_terminal("Original rank follower terminal could not acquire its published admission or committed frontier");
        }
        if (!handoff.borrowed() || handoff.frontierStream() != stream ||
            handoff.borrowerRole() != DeviceTimelineRole::DeviceGenerationController || handoff.requestCount() != 1)
            return fail_terminal("Original rank follower terminal does not own its exact retained controller stream and request");
        if (!stage.submitRetainedStorageTerminal())
            return fail_terminal("Original rank follower terminal could not submit its retained storage close");
        if (!stage.publishDeviceGenerationStateReady(stream, 1, DeviceGenerationStatePublicationKind::Terminal))
            return fail_terminal("Original rank follower terminal could not publish its completed controller frontier");
        if (!stage.consumeDeviceGenerationStateReady(stream, DeviceTimelineRole::HostResultBridge,
                1, "original_rank_follower_terminal"))
            return fail_terminal("Original rank follower terminal could not acquire its final host-result event");
        if (!backend->deviceToHostOnStream(&observations[i].cached_tokens,
                stage.state_.kv_cache->deviceSequenceCachedTokenCountPtr(0), sizeof(int),
                stage.state_.device_id.ordinal, stream) ||
            !observeQwenSparseFollowerCounters(QwenSparseTerminalOwner::PipelineFollower,
                {stage.device_generation_storage_.controlForRequest(0), sampling_math::kDeviceGenerationControlCount},
                observations[i].counters, [&](void *destination, const void *source, std::size_t bytes) {
                    return backend->deviceToHostOnStream(destination, source, bytes,
                        stage.state_.device_id.ordinal, stream);
                }))
        {
            // A failed native copy may have accepted earlier copies. Their
            // stack backing cannot be released without a native terminal proof.
            LOG_ERROR("Original rank follower compact terminal copy failed after native submission");
            std::terminate();
        }
        receipt = {.stream = stream, .workspace = stage.workspaceGeneration(stage.state_.device_id),
            .session = stage.session_epoch_, .role = MoEOverlayInferenceGraphRole::MainDecode};
    }
    // All final copies have been submitted before querying one exact native
    // event. The immutable source terminal count is never taken from scheduler
    // cadence, nor used to repair any participant's independently owned KV.
    if (!joinOriginalRankTransaction(error))
    {
        LOG_ERROR("Original rank follower lost native completion of terminal host backing");
        std::terminate();
    }
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        if (observations[i].cached_tokens != expected_source_cached_tokens)
        {
            if (error) *error = "Original rank follower independent KV terminal differs from the actual tail terminal";
            return false;
        }
        stage.awaitRetainedStorageTerminal();
        auto &loop = stage.deviceGenerationLoopGraph();
        if ((loop.launched && !loop.retireCompletedLaunch()) ||
            !stage.device_generation_state_ready_.handoff.retireAfterHostCompletion(1)) return false;
        stage.device_generation_storage_.active_request_count = 0;
        if (rank_admission_->depth_policy.isOrdinary() &&
            !stage.observeOrdinaryGenerationForwardCompletion(source_terminal_commit_count)) return false;
    }
    const char *names[]{"verifier_prior_header_bytes", "verifier_prior_payload_bytes",
        "verifier_candidate_payload_bytes", "verifier_accepted_payload_bytes"};
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        const auto total = observations[i].counters.totals();
        const uint64_t bytes[]{total.prior_header, total.prior_payload, total.candidate_payload, total.accepted_payload};
        for (int metric = 0; metric < 4; ++metric)
            PerfStatsCollector::addCounterWithSequence("qsa", names[metric], double(bytes[metric]),
                {bytes[metric]}, "decode", stages_[i]->state_.device_id.toString(),
                {{"authority", "original_rank_pipeline_follower_terminal"}, {"request", "0"}});
    }
    terminal_cached_tokens_ = expected_source_cached_tokens;
    rank_admission_.reset();
    rank_sampling_.reset();
    phase_ = Phase::Idle;
    return true;
}


std::optional<int> PipelineDeviceGeneration::observeOriginalRankKVTerminal(std::string *error)
{
    requireLiveParticipantBorrow();
    if (!rank_topology_ || phase_ != Phase::Idle) return std::nullopt;
    std::vector<int> cached(stages_.size(), -1);
    std::vector<void *> streams(stages_.size(), nullptr);
    // Admit every ordering edge before enqueuing the first host write. A later
    // participant's readiness failure must never release an earlier copy's
    // destination while its exact producer is still writing asynchronously.
    for (size_t index = 0; index < stages_.size(); ++index)
    {
        auto &stage = *stages_[index];
        auto *backend = getBackendFor(stage.state_.device_id);
        void *const stream = stage.explicitGPUStreamForOperation("original_rank_kv_terminal");
        if (!backend || !stream || !stage.waitForLiveInferenceStateReadyForObservation(stream,
                "original_rank_kv_terminal", DeviceTimelineRole::HostResultBridge)) return std::nullopt;
        streams[index] = stream;
    }
    for (size_t index = 0; index < stages_.size(); ++index)
    {
        auto &stage = *stages_[index];
        auto *backend = getBackendFor(stage.state_.device_id);
        void *const stream = streams[index];
        if (!backend->deviceToHostOnStream(&cached[index],
                stage.state_.kv_cache->deviceSequenceCachedTokenCountPtr(0), sizeof(int),
                stage.state_.device_id.ordinal, stream))
        {
            LOG_ERROR("Original rank KV terminal copy failed after native submission");
            std::terminate();
        }
        rank_transaction_terminals_[index] = {.stream = stream,
            .workspace = stage.workspaceGeneration(stage.state_.device_id), .session = stage.session_epoch_,
            .role = MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken};
    }
    // Every destination remains alive until all exact original events complete.
    if (!joinOriginalRankTransaction(error)) std::terminate();
    if (cached.empty() || cached.front() < 0 ||
        std::any_of(cached.begin(), cached.end(), [&](int count) { return count != cached.front(); }))
    {
        if (error) *error = "Original rank standalone-condition local KV terminals disagree";
        return std::nullopt;
    }
    terminal_cached_tokens_ = cached.front();
    return terminal_cached_tokens_;
}

void PipelineDeviceGeneration::bindOriginalRankTransactionCallbacks(
    std::function<bool(MoEOverlayInferenceGraphRole, int, int)> publish,
    std::function<bool(uint64_t)> retire)
{
    requireLiveParticipantBorrow();
    if (!rank_topology_ || !hasLocalTail() || phase_ != Phase::Idle || !publish || !retire ||
        rank_publish_transaction_ || rank_retire_transaction_)
        throw std::invalid_argument("Original rank transaction publication requires one cold source owner");
    rank_publish_transaction_ = std::move(publish);
    rank_retire_transaction_ = std::move(retire);
}

uint64_t PipelineDeviceGeneration::originalRankRequestGeneration() const
{
    requireLiveParticipantBorrow();
    if (!rank_topology_ || !hasLocalTail())
        throw std::logic_error("Only the real original terminal owns request generation");
    return stages_[terminalIndex()]->session_epoch_;
}

MoEOverlayInferenceCommandIdentity PipelineDeviceGeneration::originalRankCommandIdentity(uint64_t command_id) const
{
    requireLiveParticipantBorrow();
    using Kind = MoEOverlayInferencePlacementIdentityKind;
    MoEOverlayInferenceCommandIdentity command{
        .request_generation = originalRankRequestGeneration(), .command_id = command_id,
        .placement_identity_kind = Kind::RetainedGraph};
    // This is the same original terminal participant that owns request reset,
    // sampling and generation admission. Its placement version orders source
    // commands; it is not a combined identity for unrelated stage controllers.
    // Every participant separately retains its actual graph/workspace and
    // session in RankTransactionCompletion and its original native capture.
    const auto &source = *stages_[terminalIndex()];
    if (!source.graph_builder_)
        throw std::logic_error("Original generation command lost its source graph placement owner");
    const auto &config = source.graph_builder_->config();
    if (config.isMoE() && !source.usesGraphStableMoEOverlayResidency())
    {
        // A real controller's canonical initial version is zero, not missing.
        // Installed masks/replicas likewise belong to this DGO's existing
        // placement projection; no cross-stage maximum or new counter is made.
        if (source.moe_rebalance_controller_ || !source.current_expert_masks_.empty() ||
            source.current_expert_replica_set_.hasLayerReplicaPlacement())
        {
            command.placement_identity_kind = Kind::ControllerPlacementEpoch;
            command.initial_placement_epoch = source.moePlacementEpoch();
        }
        // Without a controller, the cold factory validator above has already
        // proven either disabled maintenance or the real singleton contract.
        // Their immutable graph is authoritative even if Dynamic was requested:
        // the factory deliberately creates no writer for a singleton.
    }
    if (!command.valid())
        throw std::logic_error("Original generation command has an invalid source request, command or placement identity");
    return command;
}

bool PipelineDeviceGeneration::completeOriginalRankSourceTransaction(
    MoEOverlayInferenceGraphRole role, std::string *error)
{
    requireLiveParticipantBorrow();
    if (!rank_topology_ || !hasLocalTail()) return false;
    for (size_t i = 0; i < stages_.size(); ++i)
    {
        auto &stage = *stages_[i];
        auto &receipt = rank_transaction_terminals_[i];
        if (receipt.recorded || receipt.stream) return false;
        void *stream = nullptr;
        if (role == MoEOverlayInferenceGraphRole::MTPGroupedVerifier)
        {
            const auto &publication = stage.accepted_spec_publication_ready_;
            if (!publication.valid || !publication.event || !publication.producer_stream) return false;
            stream = publication.producer_stream;
        }
        else
        {
            const auto forward = stage.forward_engine_ ? stage.forward_engine_->lastExecutedForwardGraph() : std::nullopt;
            if (!forward || !*forward || !forward->stream) return false;
            stream = forward->stream;
        }
        receipt = {.stream = stream, .workspace = stage.workspaceGeneration(stage.state_.device_id),
            .session = stage.session_epoch_, .role = role};
    }
    auto completion = recordOriginalRankTransaction(error);
    if (!completion) return false;
    rank_source_completions_.push_back(std::move(completion));
    return true;
}

}
