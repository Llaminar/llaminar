/**
 * @file PipelineForwardGraphEdges.cpp
 * @brief Native and cross-backend nodes inside complete captured pipeline forwards.
 *
 * Each edge transfers the exact physical rows, not the arena's capacity. The
 * prefill materializer or MTP row preparation separately owns real-row
 * counts, so padded transport never advances KV. Native stream order and
 * send/receive dependencies protect bank reuse without a host stage walk or
 * blocking stream wait. Accepted-state edges likewise broadcast only the tail's
 * committed four-word result. This contributor owns neither sampling nor the
 * acceptance decision; it orders each participant's existing publication stage.
 */
#include "PipelineForwardGraphEdges.h"
#include "PipelineActivationExchange.h"
#include "PipelineMetadataExchange.h"
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
    ComputeGraph graph;
    DeviceGraphExecutor::GraphSegmentCache cache;
};

bool PipelineForwardGraphEdges::PublicationBanks::valid() const noexcept
{
    const std::array fields{restore_rows, cached_tokens, accepted_counts, healthy};
    for (size_t i = 0; i < fields.size(); ++i)
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
    return {params.accepted_restore_rows_device, params.target_cached_tokens_device,
        params.accepted_state_counts_device, params.publication_ok_flags_device};
}

PipelineForwardGraphEdges::~PipelineForwardGraphEdges() = default;

std::shared_ptr<PipelinePublicationTransport> PipelineForwardGraphEdges::materializePublicationTransport(PublicationBanks banks,
    DeviceGraphExecutor &executor, IDeviceContext &context, IWorkerGPUContext &gpu,
    std::vector<PipelineMetadataBank> retained_banks)
{
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
    std::unique_ptr<PipelineMetadataExchange> exchange;
    if (domain_)
    {
        const std::array<void *, 4> expected{banks.restore_rows, banks.cached_tokens, banks.accepted_counts, banks.healthy};
        if (retained_banks.size() != expected.size())
            throw std::invalid_argument("Pipeline domain publication requires every original storage owner");
        for (size_t i = 0; i < expected.size(); ++i)
            if (retained_banks[i].data() != expected[i] || retained_banks[i].elements() != 1)
                throw std::invalid_argument("Pipeline domain publication changed its committed bank identity");
        exchange = std::make_unique<PipelineMetadataExchange>(device_,
            PipelineMetadataLayout(PipelineMetadataLayout::Kind::CommittedState, 1),
            PipelineMetadataExchange::CapturedDomain{context_, participant_, domain_->metadata_in, domain_->metadata_out},
            std::move(retained_banks), *transfer_);
    }
    else
    {
        if (!retained_banks.empty())
            throw std::invalid_argument("Native pipeline publication cannot install a second storage convention");
        exchange = nativeCommit(*context_, participant_, device_, banks);
    }
    transport->graph.addNode("pipeline_mtp_committed_state", std::move(exchange), device_);
    transport->graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
    if (publicationOrder() != PublicationOrder::LocalCommit)
    {
        if (!transport->cache.ensureCaptureStream(&gpu, device_, /*context_from_process_pool=*/true)) return {};
        transport->cache.perf_context = "pipeline_mtp_committed_state";
        DeviceGraphExecutor::DecodeCapturePolicy policy;
        policy.allow_cached_graph_replay = true;
        policy.collectives_graph_capturable = true;
        policy.defer_final_sync = true;
        // A domain with incoming words broadcasts them to all native members.
        // The terminal leader's outgoing-only channel has no native collective:
        // its siblings commit locally and must not join a fictitious capture.
        if (!domain_ || domain_->metadata_in)
            policy.capture_boundary = [this](const std::string &boundary, void *stream) {
                return captureBoundary("mtp_publication_transport:" + boundary, stream);
            };
        if (!executor.executeDecodeWithCapturePolicy(transport->graph, &context, &transport->cache,
                transport->cache.capture_stream, &gpu, nullptr, policy, nullptr,
                DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)) return {};
    }
    publication_banks_ = banks;
    publication_transport_ = transport;
    return transport;
}

const IGPUGraphCapture *PipelineForwardGraphEdges::publicationTransport(PublicationBanks banks) const
{
    const auto transport = publication_transport_.lock();
    if (!transport || transport->banks != banks) return nullptr;
    const auto view = transport->cache.deviceLoopGraphTemplate(transport->graph);
    return view ? view->capture : nullptr;
}

std::optional<PipelineForwardGraphEdges::PublicationOrder> PipelineForwardGraphEdges::preparedPublication(
    PublicationBanks banks) const
{
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
    : context_(domain.context), participant_(domain.participant),
      hidden_(*(dynamic_cast<TensorBase *>(arena.getTensor(BufferId::HIDDEN_STATE))
          ? dynamic_cast<TensorBase *>(arena.getTensor(BufferId::HIDDEN_STATE))
          : throw std::invalid_argument("Pipeline domain has no local hidden arena owner"))),
      width_(width), device_(device), membership_(device, context_, participant_),
      follower_(follower), domain_(std::move(domain)), arena_(&arena), transfer_(&transfer)
{
    const auto &port = *domain_;
    if (!device_.is_gpu() || width_ <= 0 || hidden_.native_type() != TensorType::FP32 ||
        port.stage_count < 2 || port.stage_index >= port.stage_count ||
        bool(port.activation_in) != (port.stage_index > 0) ||
        bool(port.activation_out) != (port.stage_index + 1 < port.stage_count) ||
        bool(port.metadata_in) != bool(port.activation_out) ||
        bool(port.metadata_out) != bool(port.activation_in) ||
        (follower_ && (!follower_->valid() || terminal())))
        throw std::invalid_argument("Pipeline domain has incomplete directed edges, geometry or state ownership");
    // Constructing the activation stages validates exact native membership and
    // leader ownership now, before any model graph or metadata graph is built.
    const auto bank = arena.getSharedTensor(BufferId::HIDDEN_STATE);
    if (!bank || bank.get() != &hidden_) throw std::invalid_argument("Pipeline hidden bank has no retained arena identity");
    if (receivesActivation())
        (void)PipelineActivationExchange(device_, bank, {context_, participant_, port.activation_in,
            {1, sizeof(float)}, CapturedTransferEndpoint::Consumer}, transfer);
    if (sendsActivation())
        (void)PipelineActivationExchange(device_, bank, {context_, participant_, port.activation_out,
            {1, sizeof(float)}, CapturedTransferEndpoint::Producer}, transfer);
}

bool PipelineForwardGraphEdges::terminal() const noexcept
{ return domain_ ? domain_->stage_index + 1 == domain_->stage_count : participant_ + 1 == context_->degree(); }

bool PipelineForwardGraphEdges::receivesActivation() const noexcept
{ return domain_ ? domain_->stage_index > 0 : participant_ > 0; }

bool PipelineForwardGraphEdges::sendsActivation() const noexcept
{ return domain_ ? bool(domain_->activation_out) && participant_ == 0 : participant_ + 1 < context_->degree(); }

PipelineForwardGraphEdges::PublicationOrder PipelineForwardGraphEdges::publicationOrder() const noexcept
{
    if (domain_ && !domain_->metadata_in && participant_ != 0) return PublicationOrder::LocalCommit;
    return terminal() ? PublicationOrder::CommitThenExchange : PublicationOrder::ExchangeThenCommit;
}

bool PipelineForwardGraphEdges::encloses(const ForwardInput &input) const noexcept
{
    // Select from semantic ownership, never M or the request's global MTP flag.
    // Validation in append rejects contradictory fields instead of interpreting
    // an incomplete verifier as an ordinary graph with no activation receive.
    return input.device_prefill_chunk.has_value() ||
           input.execution_role == ForwardExecutionRole::GroupedMTPVerifier ||
           input.execution_role == ForwardExecutionRole::MTPCondition ||
           (domain_ && input.execution_role == ForwardExecutionRole::MainInference &&
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
    banks.push_back(arenaBank(verifier_bank ? BufferId::MTP_VERIFIER_INPUT_TOKENS : BufferId::MTP_LOGICAL_SEQUENCE_STATE,
        input.token_ids_device, layout.elements(0)));
    if (kind != Kind::NextToken)
    {
        banks.push_back(arenaBank(verifier_bank ? BufferId::MTP_VERIFIER_POSITION_IDS : BufferId::MTP_LOGICAL_SEQUENCE_STATE,
            input.position_ids_device, layout.elements(1)));
        banks.push_back(arenaBank(verifier_bank ? BufferId::MTP_VERIFIER_REQUEST_LENGTHS : BufferId::MTP_LOGICAL_SEQUENCE_STATE,
            input.sequence_lengths_device, layout.elements(2)));
    }
    return std::make_unique<PipelineMetadataExchange>(device_, layout,
        PipelineMetadataExchange::CapturedDomain{context_, participant_, domain_->metadata_in, domain_->metadata_out},
        std::move(banks), *transfer_);
}

void PipelineForwardGraphEdges::validateInput(const ForwardInput &input) const
{
    membership_.validate();
    const bool prefill = input.execution_role == ForwardExecutionRole::MainInference &&
        input.execution_phase == ForwardExecutionPhase::Prefill && input.device_prefill_chunk.has_value();
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
    const bool ordinary = domain_ && input.execution_role == ForwardExecutionRole::MainInference &&
        input.execution_phase == ForwardExecutionPhase::Decode && input.seq_len == 1 &&
        input.token_ids_device && input.position_ids_device && input.device_decode_position &&
        !input.device_prefill_chunk && !input.token_ids && !input.position_ids &&
        input.position_policy == ForwardPositionPolicy::ExplicitRows &&
        input.state_transaction == ForwardStateTransaction::Ordinary;
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
    if (input.device != device_ || input.seq_len <= 0 || input.batch_size != 1 ||
        (!prefill && !verifier && !condition && !ordinary) ||
        (verifier && !tail && (!follower_ || input.kv_cache != follower_->checkpoint.cache)) ||
        input.external_hidden_state != (receivesActivation() ? &hidden_ : nullptr) ||
        size_t(input.seq_len) > hidden_.numel() / size_t(width_))
        throw std::invalid_argument("Pipeline forward changed its exact participant, bank, semantic role or physical row geometry");
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
        return std::make_unique<PipelineActivationExchange>(device_, arena_->getSharedTensor(BufferId::HIDDEN_STATE),
            PipelineActivationExchange::CapturedDomain{context_, participant_, endpoint == CapturedTransferEndpoint::Producer
                ? domain_->activation_out : domain_->activation_in,
                {0x4143540000000000ull | uint64_t(input.seq_len), elements * sizeof(float)}, endpoint}, *transfer_);
    };
    auto receive_stage = receivesActivation() ? activation(CapturedTransferEndpoint::Consumer) : nullptr;
    auto send_stage = sendsActivation() ? activation(CapturedTransferEndpoint::Producer) : nullptr;
    auto metadata_stage = verifier || condition || (domain_ && !input.device_prefill_chunk)
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
    if (receive_stage)
    {
        constexpr const char *receive = "pipeline_activation_receive";
        graph.addNode(receive, std::move(receive_stage), device_);
        if (!preparation.empty()) graph.addDependency(receive, preparation);
        for (const auto &root : roots) graph.addDependency(root, receive);
    }
    else if (!preparation.empty())
        for (const auto &root : roots) graph.addDependency(root, preparation);
    if (send_stage)
    {
        constexpr const char *send = "pipeline_activation_send";
        graph.addNode(send, std::move(send_stage), device_);
        for (const auto &leaf : leaves) graph.addDependency(send, leaf);
        graph.setTerminalNode(send);
    }
    if (domain_)
    {
        // The leader records the inter-vendor send; its siblings do not. The
        // collective rendezvous therefore names the shared logical forward,
        // not participant-local first/last stage names. One existing active
        // wave contract covers the unannotated ingress/egress in this capture;
        // it adds no kernel, barrier or runtime host coordination.
        const auto identity = "pipeline_domain_" + std::to_string(domain_->stage_index) +
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
    return stream && (!context_ || context_->graphCaptureBoundaryOnStream("pipeline_forward:" + boundary,
        participant_, stream, collective_timeout_policy::effectiveCollectTimeoutMs(0)));
}
}
