/**
 * @file PipelineForwardGraphEdges.cpp
 * @brief Native and cross-backend nodes inside complete captured pipeline forwards.
 *
 * Each edge transfers only original useful rows. The prefill materializer's
 * retained signed row input lowers to live bytes in channel acquisition and
 * authenticates completed TP fanout before model/KV roots. Fixed MTP/ordinary
 * metadata retains its original exact-row protocol. Native stream order and
 * send/receive dependencies protect bank reuse without a host stage walk or
 * blocking stream wait. Accepted-state edges broadcast the tail's four committed
 * state fields and, where required, three existing committed-round fields.
 * This contributor owns neither sampling nor the
 * acceptance decision; it orders each participant's existing publication stage.
 */
#include "PipelineForwardGraphEdges.h"
#include "Qwen4ExpFixedHeadPipelinePort.h"
#include "PipelineActivationExchange.h"
#include "PipelineActivationFanout.h"
#include "PipelineMetadataExchange.h"
#include "backends/NativeContextPreparation.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "kernels/common/SamplingMath.h"
#include "collective/LocalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "memory/StageBufferContract.h"
#include "memory/BufferArena.h"
#include "tensors/TensorClasses.h"
#include <array>
#include <stdexcept>
#include <sstream>

namespace llaminar2
{
namespace
{
/** @brief Bind resident rows using their semantic condition/verifier identity. */
std::unique_ptr<PipelineMetadataExchange> nativeRows(ILocalTPContext &context,
    int participant, const ForwardInput &input)
{
    const std::array<void *, 3> fields{const_cast<void *>(input.token_ids_device),
        const_cast<void *>(input.position_ids_device), const_cast<int32_t *>(input.sequence_lengths_device)};
    const auto kind = input.execution_role == ForwardExecutionRole::GroupedMTPVerifier
        ? PipelineMetadataLayout::Kind::Verifier : PipelineMetadataLayout::Kind::Condition;
    return std::make_unique<PipelineMetadataExchange>(input.device, PipelineMetadataLayout(kind, input.seq_len),
        PipelineMetadataExchange::Native{context, participant}, fields);
}

/** @brief Bind only the terminal's four committed banks, not its sampler/controller. */
std::unique_ptr<PipelineMetadataExchange> nativeCommit(ILocalTPContext &context,
    int participant, DeviceId device, PipelineForwardGraphEdges::PublicationBanks banks)
{
    const std::array<void *, 4> fields{banks.restore_rows, banks.cached_tokens, banks.accepted_counts, banks.healthy};
    return std::make_unique<PipelineMetadataExchange>(device,
        PipelineMetadataLayout(PipelineMetadataLayout::Kind::CommittedState, 1),
        PipelineMetadataExchange::Native{context, participant}, fields);
}

}

/** @brief Own only the immutable wire capture; local publication parents borrow it. */
struct PipelinePublicationTransport
{
    PipelineForwardGraphEdges::PublicationBanks banks;
    std::vector<PipelineMetadataBank> local_banks; ///< Exact retained owners for a local-only commit.
    std::optional<ComputeGraph> graph; ///< Engaged only for an actual metadata transport endpoint.
    DeviceGraphExecutor::GraphSegmentCache cache;
};

PipelineForwardGraphEdges::NativePublicationCaptureWave::NativePublicationCaptureWave(
    ILocalTPContext &context, int participant)
    : context_(context), participant_(participant)
{
    if (!context_.isLocal() || context_.degree() < 2 ||
        context_.devices().size() != static_cast<size_t>(context_.degree()) ||
        participant_ < 0 || participant_ >= context_.degree() ||
        (context_.backend() != CollectiveBackendType::NCCL && context_.backend() != CollectiveBackendType::RCCL))
        throw std::invalid_argument("Pipeline publication capture wave lost its original local native member");
}

bool PipelineForwardGraphEdges::NativePublicationCaptureWave::arrive(State phase, void *stream)
{
    if (state_ != phase || (phase != State::Begin && phase != State::End) ||
        reinterpret_cast<std::uintptr_t>(stream) <= 2 || (stream_ && stream_ != stream))
    {
        state_ = State::Failed;
        return false;
    }
    const char *const boundary = phase == State::Begin
        ? "pipeline_forward:mtp_publication_transport:native_capture_begin"
        : "pipeline_forward:mtp_publication_transport:native_capture_end";
    if (!context_.graphCaptureBoundaryOnStream(boundary, participant_, stream,
            collective_timeout_policy::effectiveCollectTimeoutMs(0)))
    {
        state_ = State::Failed;
        return false;
    }
    stream_ = stream;
    state_ = phase == State::Begin ? State::End : State::Complete;
    return true;
}

bool PipelineForwardGraphEdges::NativePublicationCaptureWave::activeBoundary(
    std::string_view boundary, void *stream)
{
    // Translate the executor's existing cold boundary hook into this one
    // publication wave. Both active failure and success must retire capture
    // before passive siblings can leave the original end rendezvous.
    if (boundary.ends_with(":context=pipeline_mtp_committed_state"))
    {
        if (boundary.starts_with("before_begin:phase=capture_begin:"))
            return arrive(State::Begin, stream);
        if (boundary.starts_with("before_begin:phase=capture_end:"))
            return arrive(State::End, stream);
        if (boundary.starts_with("before_begin:phase=capture_end_failed:"))
        {
            const bool joined = arrive(State::End, stream);
            state_ = State::Failed;
            return joined;
        }
    }
    state_ = State::Failed;
    return false;
}

bool PipelineForwardGraphEdges::NativePublicationCaptureWave::joinPassive(void *stream)
{
    return arrive(State::Begin, stream) && arrive(State::End, stream);
}

bool PipelineForwardGraphEdges::requiresMaintenanceAdmission(ForwardExecutionRole role) const noexcept
{
    return domain_ && domain_->maintenance_admission && !terminal() &&
        role == ForwardExecutionRole::GroupedMTPVerifier;
}

bool PipelineForwardGraphEdges::PublicationBanks::valid() const noexcept
{
    const std::array fields{restore_rows, cached_tokens, accepted_counts, healthy,
        committed_round_evidence[0], committed_round_evidence[1], committed_round_evidence[2]};
    const bool evidence = committed_round_evidence != std::array<int32_t *, 3>{};
    for (size_t i = 0; i < (evidence ? fields.size() : 4u); ++i)
    {
        if (!fields[i]) return false;
        for (size_t j = i + 1; j < fields.size(); ++j)
            if (fields[i] == fields[j]) return false;
    }
    return true;
}

PipelineForwardGraphEdges::PublicationBanks PipelineForwardGraphEdges::PublicationBanks::from(
    const MTPSpeculativeStatePublicationStage::Params &params) noexcept
{
    PublicationBanks banks{params.accepted_restore_rows_device.data(), params.target_cached_tokens_device,
        params.accepted_state_counts_device, params.publication_ok_flags_device};
    if (params.pipeline_moe_commit_evidence && params.pipeline_moe_commit_evidence->valid()) {
        auto *meta = params.pipeline_moe_commit_evidence->meta_device;
        banks.committed_round_evidence = {meta + sampling_math::kSpecBatchMetaOk,
            meta + sampling_math::kSpecBatchMetaOutputCount, meta + sampling_math::kSpecBatchMetaLeadingCommittedOutputCount};
    }
    return banks;
}

PipelineForwardGraphEdges::~PipelineForwardGraphEdges() = default;

std::shared_ptr<PipelinePublicationTransport> PipelineForwardGraphEdges::materializePublicationTransport(PublicationBanks banks,
    DeviceGraphExecutor &executor, IDeviceContext &context, IWorkerGPUContext &gpu,
    void *original_setup_stream, std::vector<PipelineMetadataBank> retained_banks)
{
    if (rank_forward_only_)
        throw std::invalid_argument("Forward-only rank ports do not own generation/MTP publication");
    if (!banks.valid() || context.deviceId() != device_)
        throw std::invalid_argument("Pipeline publication transport requires four distinct local metadata banks");
    if (publication_banks_)
    {
        if (*publication_banks_ != banks)
            throw std::logic_error("Frozen pipeline publication transport cannot rebind its metadata banks");
        auto existing = publication_transport_.lock();
        if (!existing || !preparedPublication(banks))
            throw std::logic_error("Retired pipeline publication transport cannot be resurrected");
        return existing;
    }
    // Only this setup wave records NCCL/RCCL work. Local acceptance policy
    // changes later compose this recording, never re-enter its rendezvous.
    auto transport = std::make_shared<PipelinePublicationTransport>();
    transport->banks = banks;
    const auto order = publicationOrder();
    std::optional<NativePublicationCaptureWave> native_wave;
    if (domain_ && domain_->native_context)
        native_wave.emplace(*domain_->native_context, domain_->native_participant);
    std::unique_ptr<PipelineMetadataExchange> exchange;
    if (domain_)
    {
        std::vector<void *> expected{banks.restore_rows, banks.cached_tokens, banks.accepted_counts, banks.healthy};
        if (carriesMoECommitEvidence()) expected.insert(expected.end(), banks.committed_round_evidence.begin(), banks.committed_round_evidence.end());
        if (carriesMoECommitEvidence() != (banks.committed_round_evidence != std::array<int32_t *, 3>{}))
            throw std::invalid_argument("Pipeline publication changed its admitted committed-round consumer schema");
        if (retained_banks.size() != expected.size())
            throw std::invalid_argument("Pipeline domain publication requires every original storage owner");
        for (size_t i = 0; i < expected.size(); ++i)
            if (retained_banks[i].device() != device_ || retained_banks[i].data() != expected[i] ||
                retained_banks[i].elements() != 1)
                throw std::invalid_argument("Pipeline domain publication changed its committed bank identity");
        if (order == PublicationOrder::LocalCommit)
        {
            transport->local_banks = std::move(retained_banks);
            if (native_wave && !native_wave->joinPassive(original_setup_stream)) return {};
        }
        else
        {
            exchange = std::make_unique<PipelineMetadataExchange>(device_,
                PipelineMetadataLayout(carriesMoECommitEvidence() ? PipelineMetadataLayout::Kind::CommittedStateWithMoERounds :
                    PipelineMetadataLayout::Kind::CommittedState, 1),
                PipelineMetadataExchange::CapturedDomain{context_, participant_, domain_->metadata_in, domain_->metadata_out,
                    domain_->native_metadata_in, domain_->native_metadata_out, domain_->native_participant},
                std::move(retained_banks), *transfer_);
        }
    }
    else
    {
        if (!retained_banks.empty())
            throw std::invalid_argument("Native pipeline publication cannot install a second storage convention");
        exchange = nativeCommit(*context_, participant_, device_, banks);
    }
    if (order != PublicationOrder::LocalCommit)
    {
        auto &graph = transport->graph.emplace();
        graph.addNode("pipeline_mtp_committed_state", std::move(exchange), device_);
        graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        if (!transport->cache.ensureCaptureStream(&gpu, device_, /*context_from_process_pool=*/true)) return {};
        transport->cache.perf_context = "pipeline_mtp_committed_state";
        DeviceGraphExecutor::DecodeCapturePolicy policy;
        policy.allow_cached_graph_replay = true;
        policy.collectives_graph_capturable = true;
        policy.defer_final_sync = true;
        if (native_wave)
        {
            // Only the real wire endpoints record this unit. The original
            // cohort still joins its two cold phases, including local commits.
            policy.graph_replay_plan_policy = DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
            policy.capture_boundary = [&](const std::string &boundary, void *stream) {
                return stream == transport->cache.capture_stream && native_wave->activeBoundary(boundary, stream);
            };
        }
        else if (!domain_ || domain_->metadata_in)
        {
            // Existing mapped receives fan out inside their own stage context.
            policy.capture_boundary = [this](const std::string &boundary, void *stream) {
                return captureBoundary("mtp_publication_transport:" + boundary, stream);
            };
        }
        if (!executor.executeDecodeWithCapturePolicy(graph, &context, &transport->cache,
                transport->cache.capture_stream, &gpu, nullptr, policy, nullptr,
                DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch) ||
            (native_wave && !native_wave->complete())) return {};
    }
    publication_banks_ = banks;
    publication_transport_ = transport;
    return transport;
}

const IGPUGraphCapture *PipelineForwardGraphEdges::publicationTransport(PublicationBanks banks) const
{
    const auto transport = publication_transport_.lock();
    if (!transport || transport->banks != banks || !transport->graph) return nullptr;
    const auto view = transport->cache.deviceLoopGraphTemplate(*transport->graph);
    return view ? view->capture : nullptr;
}

std::optional<PipelineForwardGraphEdges::PublicationOrder> PipelineForwardGraphEdges::preparedPublication(
    PublicationBanks banks) const
{
    if (rank_forward_only_) return std::nullopt;
    const auto retained = publication_transport_.lock();
    if (!retained || retained->banks != banks) return std::nullopt;
    const auto order = publicationOrder();
    if (order != PublicationOrder::LocalCommit && !publicationTransport(banks)) return std::nullopt;
    return order;
}

PipelineForwardGraphEdges::PipelineForwardGraphEdges(ILocalTPContext &context, int participant,
    TensorBase &hidden, int width, std::optional<FollowerState> follower)
    : context_(&context), participant_(participant), hidden_(hidden), width_(width),
      device_(participant >= 0 && participant < context.degree()
          ? context.devices().at(participant).toLocalDeviceId() : DeviceId::cpu()),
      membership_(device_, &context, participant), follower_(follower)
{
    if (context.degree() < 2 || participant < 0 || participant >= context.degree() ||
        !device_.is_gpu() || width <= 0 || hidden.native_type() != TensorType::FP32 ||
        (context.backend() != CollectiveBackendType::NCCL && context.backend() != CollectiveBackendType::RCCL))
        throw std::invalid_argument("Pipeline forward requires a frozen native GPU context and local FP32 bank");
    if (follower_ && (!follower_->valid() || participant + 1 == context.degree()))
        throw std::invalid_argument("Pipeline follower checkpoint must be complete and belong to a nonterminal participant");
}

PipelineForwardGraphEdges::PipelineForwardGraphEdges(DeviceId device, CapturedDomain domain,
    BufferArena &arena, int width, std::optional<FollowerState> follower, TransferEngine &transfer)
    : PipelineForwardGraphEdges(device, std::move(domain), arena, width, follower, transfer,
        BoundaryCoverage::CompleteGeneration) {}

PipelineForwardGraphEdges::PipelineForwardGraphEdges(DeviceId device, RankBoundary boundary,
    BufferArena &arena, int width, TransferEngine &transfer)
    : PipelineForwardGraphEdges(device, CapturedDomain{boundary.stage_index, boundary.stage_count,
        boundary.context, boundary.participant, std::move(boundary.activation_in),
        std::move(boundary.activation_out), nullptr, nullptr, std::move(boundary.activation_fanout)},
        arena, width, std::nullopt, transfer, BoundaryCoverage::ForwardOnly) {}

PipelineForwardGraphEdges::PipelineForwardGraphEdges(DeviceId device, CapturedDomain domain,
    BufferArena &arena, int width, std::optional<FollowerState> follower, TransferEngine &transfer,
    BoundaryCoverage coverage)
    : context_(domain.context), participant_(domain.participant),
      hidden_(*(dynamic_cast<TensorBase *>(arena.getTensor(BufferId::HIDDEN_STATE))
          ? dynamic_cast<TensorBase *>(arena.getTensor(BufferId::HIDDEN_STATE))
          : throw std::invalid_argument("Pipeline domain has no local hidden arena owner"))),
      width_(width), device_(device), membership_(device, context_, participant_),
      follower_(follower), domain_(std::move(domain)), rank_forward_only_(coverage == BoundaryCoverage::ForwardOnly),
      arena_(&arena), transfer_(&transfer)
{
    const auto &port = *domain_;
    if (!device_.is_gpu() || width_ <= 0 || hidden_.native_type() != TensorType::FP32 ||
        port.stage_count < 2 || port.stage_index >= port.stage_count ||
        bool(port.activation_in || port.native_activation_in) != (port.stage_index > 0) ||
        bool(port.activation_out || port.native_activation_out) != (port.stage_index + 1 < port.stage_count) ||
        (rank_forward_only_ ? bool(port.metadata_in || port.native_metadata_in) || bool(port.metadata_out || port.native_metadata_out) || bool(follower_)
            : bool(port.metadata_in || port.native_metadata_in) != bool(port.activation_out || port.native_activation_out) ||
              bool(port.metadata_out || port.native_metadata_out) != bool(port.activation_in || port.native_activation_in)) ||
        (bool(port.native_context) != bool(port.native_activation_in || port.native_activation_out || port.native_metadata_in || port.native_metadata_out)) ||
        (port.native_context && (port.activation_in || port.activation_out || port.metadata_in || port.metadata_out)) ||
        (port.carries_moe_commit_evidence != bool(port.maintenance_admission)) ||
        (port.maintenance_admission && (!port.native_context || port.maintenance_admission->device(port.native_participant) != device_)) ||
        (follower_ && (!follower_->valid() || terminal())))
        throw std::invalid_argument("Pipeline domain has incomplete directed edges, geometry or state ownership");
    if (port.native_context) {
        for (const auto &edge : {port.native_activation_in, port.native_activation_out,
                port.native_metadata_in, port.native_metadata_out})
            if (edge) {
                edge->requireOriginal(*port.native_context, port.native_participant);
                if (edge->device(port.native_participant) != device_)
                    throw std::invalid_argument("Native pipeline port changed its original physical member");
            }
    }
    if (port.activation_fanout)
    {
        if (!receivesActivation() || !context_)
            throw std::invalid_argument("Pipeline fanout must belong to this receiving native TP domain");
        port.activation_fanout->requireOriginal(device_, context_, participant_, port.activation_in);
    }
    // Constructing the activation stages validates exact native membership and
    // leader ownership now, before any model graph or metadata graph is built.
    const auto bank = arena.getSharedTensor(BufferId::HIDDEN_STATE);
    if (!bank || bank.get() != &hidden_) throw std::invalid_argument("Pipeline hidden bank has no retained arena identity");
    if (receivesActivation())
        (void)PipelineActivationExchange(device_, bank, {context_, participant_, port.activation_in,
            {1, sizeof(float)}, CapturedTransferEndpoint::Consumer, std::nullopt, port.activation_fanout,
            port.native_activation_in, port.native_participant}, transfer);
    if (sendsActivation())
        (void)PipelineActivationExchange(device_, bank, {context_, participant_, port.activation_out,
            {1, sizeof(float)}, CapturedTransferEndpoint::Producer, std::nullopt, {},
            port.native_activation_out, port.native_participant}, transfer);
}

bool PipelineForwardGraphEdges::terminal() const noexcept
{ return domain_ ? domain_->stage_index + 1 == domain_->stage_count : participant_ + 1 == context_->degree(); }

std::shared_ptr<const Qwen4ExpFixedHeadPipelinePort> PipelineForwardGraphEdges::fixedHeadPLEConsumer() const noexcept
{
    return domain_ && domain_->fixed_head_port && domain_->fixed_head_port->consumesPLE()
        ? domain_->fixed_head_port : nullptr;
}
bool PipelineForwardGraphEdges::receivesActivation() const noexcept
{ return domain_ ? domain_->stage_index > 0 : participant_ > 0; }

bool PipelineForwardGraphEdges::sendsActivation() const noexcept
{ return domain_ ? bool(domain_->activation_out || domain_->native_activation_out) && participant_ == 0 : participant_ + 1 < context_->degree(); }

PipelineForwardGraphEdges::PublicationOrder PipelineForwardGraphEdges::domainPublicationOrder(
    const CapturedDomain &domain) noexcept
{
    // Every earlier domain consumes reverse metadata. Only the actual tail
    // leader publishes it, regardless of native or mapped edge transport.
    if (domain.stage_index + 1 < domain.stage_count) return PublicationOrder::ExchangeThenCommit;
    return domain.participant == 0 ? PublicationOrder::CommitThenExchange : PublicationOrder::LocalCommit;
}

PipelineForwardGraphEdges::PublicationOrder PipelineForwardGraphEdges::publicationOrder() const noexcept
{
    if (domain_) return domainPublicationOrder(*domain_);
    return terminal() ? PublicationOrder::CommitThenExchange : PublicationOrder::ExchangeThenCommit;
}

bool PipelineForwardGraphEdges::encloses(const ForwardInput &input) const
{
    if (rank_forward_only_)
    {
        validateInput(input);
        return true;
    }
    switch (input.pipeline_activation_ownership)
    {
    case ForwardPipelineActivationOwnership::ForwardGraph:
        break;
    case ForwardPipelineActivationOwnership::NativeGenerationParent:
        // Only a setup-only child may omit its local activation transport.
        // An executing request/history invocation cannot bypass the receive.
        if (domain_ || input.graph_submission_intent !=
                ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch ||
            input.execution_role != ForwardExecutionRole::MainInference ||
            input.execution_phase != ForwardExecutionPhase::Decode ||
            (input.state_transaction != ForwardStateTransaction::Ordinary &&
             input.state_transaction != ForwardStateTransaction::ForwardOnlyCondition) ||
            !input.device_decode_position || input.sequence_lengths_device)
            throw std::invalid_argument("Pipeline activation parent ownership requires an exact native resident setup child");
        validateInput(input);
        return false;
    default:
        throw std::invalid_argument("Unknown pipeline activation ownership");
    }
    // Select from semantic ownership, never M or the request's global MTP flag.
    // Validation in append rejects contradictory fields instead of interpreting
    // an incomplete verifier as an ordinary graph with no activation receive.
    return input.device_prefill_chunk.has_value() ||
           input.execution_role == ForwardExecutionRole::GroupedMTPVerifier ||
           input.execution_role == ForwardExecutionRole::MTPCondition ||
           (input.execution_role == ForwardExecutionRole::MainInference &&
            input.execution_phase == ForwardExecutionPhase::Decode && input.token_ids_device &&
            input.device_decode_position.has_value());
}

PipelineMetadataBank PipelineForwardGraphEdges::arenaBank(BufferId id, const void *pointer, size_t elements) const
{
    const auto tensor = arena_ ? arena_->getSharedTensor(id) : nullptr;
    if (!tensor || !tensor->gpu_data_ptr() || !pointer)
        throw std::invalid_argument("Pipeline metadata has no canonical arena owner");
    const auto base = reinterpret_cast<uintptr_t>(tensor->gpu_data_ptr());
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    if (address < base) throw std::invalid_argument("Pipeline metadata precedes its canonical arena region");
    return PipelineMetadataBank(tensor, id, address - base, elements);
}

std::unique_ptr<PipelineMetadataExchange> PipelineForwardGraphEdges::metadataRows(const ForwardInput &input) const
{
    if (rank_forward_only_)
        throw std::invalid_argument("Forward-only rank ports do not own reverse generation row metadata");
    if (!domain_) return nativeRows(*context_, participant_, input);
    using Kind = PipelineMetadataLayout::Kind;
    const auto kind = input.execution_role == ForwardExecutionRole::MainInference ? Kind::NextToken :
        input.execution_role == ForwardExecutionRole::GroupedMTPVerifier ? Kind::Verifier : Kind::Condition;
    const PipelineMetadataLayout layout(kind, input.seq_len);
    std::vector<PipelineMetadataBank> banks;
    const auto verifier_tokens = arena_->getSharedTensor(BufferId::MTP_VERIFIER_INPUT_TOKENS);
    const bool verifier_bank = verifier_tokens && input.token_ids_device == verifier_tokens->gpu_data_ptr();
    if (kind == Kind::Verifier && !verifier_bank)
        throw std::invalid_argument("Pipeline verifier must use its original complete row bank");
    // External scalar decode consumes the request admission bank; a resident
    // controller consumes its logical mailbox. Both retain their original
    // tensor authority. Only the NextToken family may bind the exact admitted
    // token base; condition/verifier rows retain their existing complete banks.
    const auto admitted_tokens = kind == Kind::NextToken
        ? arena_->getSharedTensor(BufferId::REQUEST_TOKEN_IDS) : nullptr;
    const bool admitted_bank = admitted_tokens && input.token_ids_device == admitted_tokens->gpu_data_ptr();
    const auto token_owner = verifier_bank ? BufferId::MTP_VERIFIER_INPUT_TOKENS :
        admitted_bank ? BufferId::REQUEST_TOKEN_IDS : BufferId::MTP_LOGICAL_SEQUENCE_STATE;
    banks.push_back(arenaBank(token_owner, input.token_ids_device, layout.elements(0)));
    if (kind != Kind::NextToken)
    {
        banks.push_back(arenaBank(verifier_bank ? BufferId::MTP_VERIFIER_POSITION_IDS : BufferId::MTP_LOGICAL_SEQUENCE_STATE,
            input.position_ids_device, layout.elements(1)));
        banks.push_back(arenaBank(verifier_bank ? BufferId::MTP_VERIFIER_REQUEST_LENGTHS : BufferId::MTP_LOGICAL_SEQUENCE_STATE,
            input.sequence_lengths_device, layout.elements(2)));
    }
    return std::make_unique<PipelineMetadataExchange>(device_, layout,
        PipelineMetadataExchange::CapturedDomain{context_, participant_, domain_->metadata_in, domain_->metadata_out,
                domain_->native_metadata_in, domain_->native_metadata_out, domain_->native_participant},
        std::move(banks), *transfer_);
}

void PipelineForwardGraphEdges::validateInput(const ForwardInput &input) const
{
    membership_.validate();
    const bool prefill = input.execution_role == ForwardExecutionRole::MainInference &&
        input.execution_phase == ForwardExecutionPhase::Prefill && input.device_prefill_chunk.has_value();
    if (rank_forward_only_ && (!prefill || input.pipeline_activation_ownership != ForwardPipelineActivationOwnership::ForwardGraph))
        throw std::invalid_argument("Forward-only rank transport is not complete ordinary generation or MTP");
    const bool resident_mtp_rows = input.execution_phase == ForwardExecutionPhase::Decode &&
        !input.device_prefill_chunk && input.token_ids == nullptr && input.position_ids == nullptr &&
        input.token_ids_device && input.position_ids_device && input.sequence_lengths_device &&
        input.position_policy == ForwardPositionPolicy::ExplicitRows &&
        !input.device_decode_position;
    const bool verifier = input.execution_role == ForwardExecutionRole::GroupedMTPVerifier &&
        resident_mtp_rows && input.seq_len >= 2 && !input.shifted_mtp_prefill &&
        input.state_transaction == ForwardStateTransaction::Ordinary;
    const bool prefix_bridge = input.state_transaction == ForwardStateTransaction::RestoredPrefixMTPDecodeBridge;
    const bool tail = terminal();
    const bool ordinary = input.execution_role == ForwardExecutionRole::MainInference &&
        input.execution_phase == ForwardExecutionPhase::Decode && input.seq_len == 1 &&
        input.token_ids_device && input.position_ids_device && input.device_decode_position &&
        !input.device_prefill_chunk && !input.token_ids && !input.position_ids &&
        input.position_policy == ForwardPositionPolicy::ExplicitRows &&
        input.device_decode_position->valid() &&
        input.position_ids_device == input.device_decode_position->position &&
        (input.state_transaction == ForwardStateTransaction::Ordinary ||
         input.state_transaction == ForwardStateTransaction::ForwardOnlyCondition);
    // Scalar conditions traverse the same layer pipeline as verification. Only
    // the terminal may append a shifted sidecar during a restored-prefix bridge;
    // followers neither require nor borrow that terminal-only state.
    const bool condition = input.execution_role == ForwardExecutionRole::MTPCondition &&
        resident_mtp_rows && input.seq_len == 1 &&
        (input.state_transaction == ForwardStateTransaction::Ordinary ||
         input.state_transaction == ForwardStateTransaction::CommittedMTPCondition || prefix_bridge) &&
        (prefix_bridge && tail
            ? input.shifted_mtp_prefill && input.shifted_mtp_prefill->executableForRequestCount(1)
            : !input.shifted_mtp_prefill);
    // Report the first violated original authority, before graph mutation or
    // collective enqueue. A generic geometry error hides distinct lifecycle
    // defects (for example an absent retained follower checkpoint).
    const auto reject = [&](const char *invariant) {
        std::ostringstream diagnostic;
        diagnostic << "Pipeline forward invariant failed: " << invariant
            << "; device=" << device_.toString() << "; input_device=" << input.device.toString()
            << "; terminal=" << tail << "; role=" << static_cast<int>(input.execution_role)
            << "; phase=" << static_cast<int>(input.execution_phase)
            << "; state=" << static_cast<int>(input.state_transaction)
            << "; rows=" << input.seq_len << "; batch=" << input.batch_size
            << "; row_capacity=" << hidden_.numel() / size_t(width_)
            << "; resident_mtp_rows=" << resident_mtp_rows
            << "; shifted_binding=" << input.shifted_mtp_prefill.has_value()
            << "; follower_checkpoint=" << follower_.has_value();
        throw std::invalid_argument(diagnostic.str());
    };
    if (input.device != device_) reject("original participant device");
    if (input.seq_len <= 0) reject("positive physical rows");
    if (input.batch_size != 1) reject("single-request pipeline bank");
    if (!prefill && !verifier && !condition && !ordinary) reject("complete semantic forward transaction");
    if (verifier && !tail && !follower_) reject("retained follower checkpoint required");
    if (verifier && !tail && input.kv_cache != follower_->checkpoint.cache)
        reject("original follower checkpoint cache");
    if (input.external_hidden_state != (receivesActivation() ? &hidden_ : nullptr))
        reject("original external hidden-state bank");
    if (size_t(input.seq_len) > hidden_.numel() / size_t(width_)) reject("physical rows exceed retained hidden bank");
}

void PipelineForwardGraphEdges::validateFollowerMTPInput(const ForwardInput &input) const
{
    if (terminal() ||
        (input.execution_role != ForwardExecutionRole::MTPCondition &&
         input.execution_role != ForwardExecutionRole::GroupedMTPVerifier))
        throw std::invalid_argument("Pipeline MTP receive authority belongs only to nonterminal main-model participants");
    validateInput(input);
}

void PipelineForwardGraphEdges::append(ComputeGraph &graph, const ForwardInput &input) const
{
    if (!encloses(input))
        throw std::invalid_argument("Native generation child cannot also own pipeline activation edges");
    validateInput(input);
    const bool verifier = input.execution_role == ForwardExecutionRole::GroupedMTPVerifier;
    const bool condition = input.execution_role == ForwardExecutionRole::MTPCondition;
    const bool tail = terminal();
    const auto roots = graph.getRootNodes();
    const auto leaves = graph.getLeafNodes();
    if (roots.empty() || leaves.empty()) throw std::invalid_argument("Pipeline forward model graph is empty");
    if (domain_)
        for (const auto &root : roots)
            if (graph.getNode(root)->graph_capture_wave)
                throw std::logic_error("Pipeline domain entry already belongs to a different capture wave");
    const size_t elements = size_t(input.seq_len) * size_t(width_);
    // Validate and bind all edges before changing the declarative DAG. A stale
    // arena region or undersized channel must not leave a half-enclosed graph.
    const auto activation = [&](CapturedTransferEndpoint endpoint) -> std::unique_ptr<PipelineActivationExchange> {
        if (!domain_)
            return std::make_unique<PipelineActivationExchange>(device_, hidden_, elements,
                PipelineActivationExchange::NativePeer{*context_, participant_, endpoint == CapturedTransferEndpoint::Producer
                    ? CollectiveP2POpKind::Send : CollectiveP2POpKind::Recv});
        std::optional<PipelineActivationExchange::DeviceRows> live;
        if (input.device_prefill_chunk)
        {
            const auto count = arena_->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
            if (!count || count->native_type() != TensorType::INT32 || count->size_bytes() < sizeof(int32_t) ||
                !count->gpu_data_ptr() || input.device_prefill_chunk->chunk_real_rows_device != count->gpu_data_ptr())
                throw std::invalid_argument("Pipeline live rows must retain the original materializer geometry bank");
            live = PipelineActivationExchange::DeviceRows{count, 0, size_t(width_) * sizeof(float)};
        }
        return std::make_unique<PipelineActivationExchange>(device_, arena_->getSharedTensor(BufferId::HIDDEN_STATE),
            PipelineActivationExchange::CapturedDomain{context_, participant_, endpoint == CapturedTransferEndpoint::Producer
                ? domain_->activation_out : domain_->activation_in,
                {0x4143540000000000ull | uint64_t(input.seq_len), elements * sizeof(float)}, endpoint,
                std::move(live), endpoint == CapturedTransferEndpoint::Consumer ? domain_->activation_fanout : nullptr,
                endpoint == CapturedTransferEndpoint::Producer ? domain_->native_activation_out : domain_->native_activation_in,
                domain_->native_participant}, *transfer_);
    };
    auto receive_stage = receivesActivation() ? activation(CapturedTransferEndpoint::Consumer) : nullptr;
    auto send_stage = sendsActivation() ? activation(CapturedTransferEndpoint::Producer) : nullptr;
    auto metadata_stage = !rank_forward_only_ && (verifier || condition || (domain_ && !input.device_prefill_chunk))
        ? metadataRows(input) : nullptr;
    std::string preparation;
    if (metadata_stage)
    {
        // These collectives precede both activation receipt and local compute.
        // A follower cannot consume stale rows while waiting for an activation.
        preparation = "pipeline_mtp_rows";
        graph.addNode(preparation, std::move(metadata_stage), device_);
        if (verifier && !tail)
        {
            MTPVerifierPreparationStage::Params params;
            params.device_id = device_;
            params.backend = follower_->backend;
            params.authority = MTPVerifierPreparationStage::Authority::PipelineFollower;
            params.request_count = 1;
            params.padded_seq_len = input.seq_len;
            params.main_kv_checkpoints = std::span(&follower_->checkpoint, 1);
            constexpr const char *checkpoint = "pipeline_verifier_checkpoint";
            graph.addNode(checkpoint, std::make_unique<MTPVerifierPreparationStage>(params), device_);
            graph.addDependency(checkpoint, preparation);
            preparation = checkpoint;
        }
    }
    const auto *fixed = domain_ && domain_->fixed_head_port && !domain_->fixed_head_port->consumesPLE()
        ? domain_->fixed_head_port.get() : nullptr;
    if (fixed && !fixed->suppliesAfterActivation()) {
        if (!receive_stage) throw std::logic_error("Downstream fixed PLE storage lost its original activation receive");
        const auto dependencies = preparation.empty() ? std::span<const std::string>{}
            : std::span<const std::string>{&preparation, 1};
        preparation = fixed->appendStorage(graph, input, dependencies);
    }
    if (receive_stage)
    {
        constexpr const char *receive = "pipeline_activation_receive";
        graph.addNode(receive, std::move(receive_stage), device_);
        if (!preparation.empty()) graph.addDependency(receive, preparation);
        for (const auto &root : roots) graph.addDependency(root, receive);
    }
    else if (!preparation.empty())
        for (const auto &root : roots) graph.addDependency(root, preparation);
    std::string activation_terminal;
    if (send_stage)
    {
        constexpr const char *send = "pipeline_activation_send";
        graph.addNode(send, std::move(send_stage), device_);
        for (const auto &leaf : leaves) graph.addDependency(send, leaf);
        graph.setTerminalNode(send);
        activation_terminal = send;
    }
    if (fixed && fixed->suppliesAfterActivation()) {
        // Earlier storage may wait for the PLE IDs only after its own model
        // activation has left. This breaks the cross-stage dependency cycle.
        const auto dependencies = activation_terminal.empty() ? std::span<const std::string>{leaves}
            : std::span<const std::string>{&activation_terminal, 1};
        graph.setTerminalNode(fixed->appendStorage(graph, input, dependencies));
    }
    if (domain_)
    {
        // The leader records the inter-vendor send; its siblings do not. The
        // collective rendezvous therefore names the shared logical forward,
        // not participant-local first/last stage names. One existing active
        // wave contract covers the unannotated ingress/egress in this capture;
        // it adds no kernel, barrier or runtime host coordination.
        const auto identity = (domain_->native_context ? std::string("pipeline_native_domains") :
            "pipeline_domain_" + std::to_string(domain_->stage_index)) +
            ":role=" + std::to_string(static_cast<int>(input.execution_role)) +
            ":phase=" + std::to_string(static_cast<int>(input.execution_phase)) +
            ":state=" + std::to_string(static_cast<int>(input.state_transaction)) +
            ":rows=" + std::to_string(input.seq_len);
        for (const auto &root : roots)
            graph.setGraphCaptureWaveContract(root, {.identity = identity});
    }
}

void PipelineForwardGraphEdges::validatePublication(
    const MTPSpeculativeStatePublicationStage::Params &params) const
{
    if (rank_forward_only_)
        throw std::invalid_argument("Forward-only rank ports cannot publish generation/MTP state");
    using Publication = MTPSpeculativeStatePublicationStage;
    const bool tail = terminal();
    const bool authority_matches = tail
        ? params.authority == Publication::Authority::CompactOutcome ||
          params.authority == Publication::Authority::BoundedGeneration
        : params.authority == Publication::Authority::PipelineFollower;
    if (params.device_id != device_ ||
        params.request_count != 1 || params.verifier_rows_per_request < 2 ||
        !authority_matches)
        throw std::invalid_argument("Pipeline MTP publication requires exact local authority/geometry");
    if (!tail && (!follower_ || params.backend != follower_->backend ||
        params.main_kv_bindings != std::vector<Publication::MainKVBinding>{{
            .cache = follower_->checkpoint.cache,
            .first_sequence_index = follower_->checkpoint.sequence_index,
            .base_checkpoint_device = follower_->checkpoint.checkpoint_device,
            .base_checkpoint_bytes = follower_->checkpoint.checkpoint_bytes}}))
        throw std::invalid_argument("Pipeline MTP publication changed the verifier's local checkpoint owner");

    // Each field is a distinct one-word persistent bank. Aliases would make
    // later broadcasts silently overwrite an earlier committed field.
    if (!PublicationBanks::from(params).valid())
        throw std::invalid_argument("Pipeline MTP committed metadata fields must be present and may not alias");
    if (!Publication(params).validate())
        throw std::invalid_argument("Pipeline MTP publication has incomplete persistent bindings");
}

DeviceGraphExecutor::RetainedParentCompositionHook PipelineForwardGraphEdges::publicationComposer(
    ComputeGraph &graph, MTPSpeculativeStatePublicationStage *stage, std::shared_ptr<TensorBase> evidence,
    const std::shared_ptr<PhysicalMemoryAuthority>& memory, uint64_t workspace_generation,
    IWorkerGPUContext &worker, void *stream) const
{
    using Hook = DeviceGraphExecutor::RetainedParentCompositionHook;
    using Publication = MTPSpeculativeStatePublicationStage;
    if (!stage || !stream || !workspace_generation)
        throw std::invalid_argument("Pipeline publication composer requires its exact local child and workspace");
    const auto *consumer = graph.getNode("mtp_speculative_state_publication");
    if (!consumer || consumer->stage.get() != stage || consumer->device != device_)
        throw std::invalid_argument("Pipeline publication composer rejected a foreign local child");
    const auto params = stage->getParams();
    validatePublication(params);
    const auto banks = PublicationBanks::from(params);
    const auto order = preparedPublication(banks);
    if (!order) throw std::logic_error("Pipeline publication has no original retained transport admission");
    auto transport = publication_transport_.lock();
    if (!transport) throw std::logic_error("Pipeline publication lost its original retained wire source");
    const auto *wire = *order == PublicationOrder::LocalCommit ? nullptr : publicationTransport(banks);
    if (*order != PublicationOrder::LocalCommit && !wire)
        throw std::logic_error("Pipeline publication has no exact captured wire child");

    Hook hook = [this, transport, params, stage, expected_graph = &graph](
        IGPUGraphCapture &destination, const ComputeGraph &actual_graph,
        std::span<const DeviceGraphExecutor::GraphSegmentCache::RetainedCaptureUnitTemplateView> units) {
        if (&actual_graph != expected_graph || !stage->hasSameCaptureIdentity(params) ||
            units.size() != 1 || !units.front().capture ||
            units.front().stage_names.size() != 1 ||
            units.front().stage_names.front() != "mtp_speculative_state_publication" ||
            units.front().capture->executionStream() != destination.executionStream() ||
            units.front().stream != destination.executionStream())
            return false;
        // Retain the original wire owner, but use the same topology lowering
        // as every publication: there is exactly one ordering implementation.
        return composePublication(destination, *units.front().capture, params);
    };

    // Pipeline publication is one cold cache identity. Sharing this already
    // frozen callable avoids rebuilding host closures or source permits during
    // request replay; the same original graph still validates every use.
    hook.closed_function_ = std::make_shared<const Hook::Function>(std::move(hook.function_));
    const bool consumes_received_rounds = params.authority == Publication::Authority::PipelineFollower &&
        params.dynamic_moe_commit_boundary && params.pipeline_moe_commit_evidence;
    if (!consumes_received_rounds) return hook;
    // This exemption belongs only to the genuine prior wire child. A tail or
    // local-only commit cannot claim that its bytes were received beforehand.
    if (*order != PublicationOrder::ExchangeThenCommit || !carriesMoECommitEvidence() ||
        !transport->graph || !evidence || !memory)
        throw std::logic_error("Pipeline cadence consumer has no preceding original metadata producer");
    const auto *node = transport->graph->getNode("pipeline_mtp_committed_state");
    const auto *exchange = node ? dynamic_cast<const PipelineMetadataExchange*>(node->stage.get()) : nullptr;
    if (!exchange) throw std::logic_error("Pipeline cadence producer is not the original metadata exchange");
    const auto received = exchange->receivedBanks();
    const auto fields = std::array{sampling_math::kSpecBatchMetaOk, sampling_math::kSpecBatchMetaOutputCount,
        sampling_math::kSpecBatchMetaLeadingCommittedOutputCount};
    if (received.size() != 7 || evidence->native_type() != TensorType::INT32 ||
        evidence->current_device() != device_ || !evidence->gpu_data_ptr() ||
        evidence->size_bytes() < sampling_math::kSpeculativeBatchMetaCount * sizeof(int32_t) ||
        params.pipeline_moe_commit_evidence->meta_device != evidence->gpu_data_ptr() ||
        params.pipeline_moe_commit_evidence->meta_stride != sampling_math::kSpeculativeBatchMetaCount)
        throw std::logic_error("Pipeline cadence producer changed the original compact bank schema");
    for (size_t i = 0; i < fields.size(); ++i)
    {
        const auto &bank = received[i + 4];
        const auto *owner = std::get_if<std::shared_ptr<TensorBase>>(&bank.owner_);
        if (!owner || owner->get() != evidence.get() || bank.arenaId() != BufferId::STOCHASTIC_BATCH_OUTPUT_META ||
            bank.device() != device_ || bank.elements() != 1 || bank.offset_ != fields[i] * sizeof(int32_t) ||
            bank.data() != banks.committed_round_evidence[i])
            throw std::logic_error("Pipeline cadence producer changed an exact original received field");
    }
    auto original = GPUDeviceContextPool::instance().retainExistingPreparedContext(device_, memory);
    if (original.worker().get() != &worker || transport->cache.gpu_ctx_ref != &worker ||
        transport->cache.capture_device != device_)
        throw std::logic_error("Pipeline cadence producer changed its original native worker");
    const auto recording = transport->cache.successful_capture_count;
    const auto source_topology = transport->graph->topologyGeneration();
    const auto consumer_topology = graph.topologyGeneration();
    const auto signature = DeviceGraphCaptureController::computeCaptureVariantSignature(graph);
    if (!recording) throw std::logic_error("Pipeline cadence producer has no completed native recording");
    const auto original_worker = original.worker();
    const auto validate = [transport, wire, original = std::move(original), memory, evidence, stage, params,
        consumer_graph = &graph, consumer_topology, signature, recording, source_topology, banks,
        source_node = node, source_stage = exchange]() {
        if (original.memory() != memory || original.worker().get() != transport->cache.gpu_ctx_ref ||
            transport->cache.successful_capture_count != recording ||
            !transport->graph || transport->graph->topologyGeneration() != source_topology ||
            consumer_graph->topologyGeneration() != consumer_topology ||
            DeviceGraphCaptureController::computeCaptureVariantSignature(*consumer_graph) != signature ||
            !stage->hasSameCaptureIdentity(params))
            throw std::logic_error("Pipeline cadence prior-child producer or consumer recording is stale");
        const auto current = transport->cache.deviceLoopGraphTemplate(*transport->graph);
        if (!current || current->capture != wire || current->stream != transport->cache.capture_stream)
            throw std::logic_error("Pipeline cadence producer changed its original native child");
        // The topology check above precedes any dereference of the cold-authenticated
        // node. Reuse its exact stage identity instead of constructing a long
        // temporary string for a named lookup on every replay.
        if (source_node->stage.get() != source_stage || source_stage->receivedBanks().size() != 7)
            throw std::logic_error("Pipeline cadence producer lost its exact received-bank authority");
        const std::array<void*, 7> expected{banks.restore_rows, banks.cached_tokens, banks.accepted_counts,
            banks.healthy, banks.committed_round_evidence[0], banks.committed_round_evidence[1],
            banks.committed_round_evidence[2]};
        for (size_t i = 0; i < expected.size(); ++i)
            if (source_stage->receivedBanks()[i].data() != expected[i])
                throw std::logic_error("Pipeline cadence producer changed captured physical storage");
    };
    hook.authority_ = std::shared_ptr<const RetainedParentInputAuthority>(new RetainedParentInputAuthority(
        original_worker, transport, wire, &graph, stage, device_, stream, recording, workspace_generation, signature, validate));
    hook.authority_->require(graph, device_, stream);
    return hook;
}

bool PipelineForwardGraphEdges::composePublication(IGPUGraphCapture &destination,
    const IGPUGraphCapture &local, const MTPSpeculativeStatePublicationStage::Params &params) const
{
    validatePublication(params);
    const auto order = preparedPublication(PublicationBanks::from(params));
    if (!order) return false;
    const GPUOrderedTimelineStep commit{.name = "local accepted-state commit", .capture = &local};
    if (*order == PublicationOrder::LocalCommit)
        return destination.buildOrderedTimelineTransaction(std::span(&commit, 1));
    const auto *transport = publicationTransport(PublicationBanks::from(params));
    if (!transport) return false;
    const GPUOrderedTimelineStep exchange{.name = "tail committed-state exchange", .capture = transport};
    const auto steps = *order == PublicationOrder::CommitThenExchange
        ? std::array{commit, exchange} : std::array{exchange, commit};
    return destination.buildOrderedTimelineTransaction(steps);
}

bool PipelineForwardGraphEdges::captureBoundary(const std::string &boundary, void *stream) const
{
    if (domain_ && domain_->native_context)
        return stream && domain_->native_context->graphCaptureBoundaryOnStream("pipeline_forward:" + boundary,
            domain_->native_participant, stream, collective_timeout_policy::effectiveCollectTimeoutMs(0));
    return stream && (!context_ || context_->graphCaptureBoundaryOnStream("pipeline_forward:" + boundary,
        participant_, stream, collective_timeout_policy::effectiveCollectTimeoutMs(0)));
}
}
