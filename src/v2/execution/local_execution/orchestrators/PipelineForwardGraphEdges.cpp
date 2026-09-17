/**
 * @file PipelineForwardGraphEdges.cpp
 * @brief Native point-to-point nodes inside complete captured pipeline forwards.
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
#include "collective/LocalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "memory/StageBufferContract.h"
#include "tensors/TensorClasses.h"
#include <array>
#include <stdexcept>

namespace llaminar2
{
namespace
{
/**
 * @brief Broadcast terminal-owned MTP banks at their explicit graph boundary.
 *
 * Row inputs and committed-state fields are independent allocations, not a wire struct. Fixed
 * descriptors are built at graph construction; replay never allocates, copies
 * through the host or examines token values. Only the terminal is a producer.
 */
class PipelineMTPMetadataCollective final : public IComputeStage
{
public:
    /** @brief Bind the exact participant-local destinations and terminal source role. */
    PipelineMTPMetadataCollective(ILocalTPContext &context, int participant, const ForwardInput &input)
        : IComputeStage(input.device), context_(context), participant_(participant),
          field_count_(3), name_("pipeline_mtp_rows")
    {
        const std::array pointers{input.token_ids_device,
            static_cast<const void *>(input.position_ids_device),
            static_cast<const void *>(input.sequence_lengths_device)};
        const std::array<const char *, 3> names{"mtp_tokens", "mtp_positions", "mtp_width"};
        for (size_t index = 0; index < field_count_; ++index)
            fields_[index] = {.kind = LocalTPCollectiveSidebandKind::Broadcast,
                .send_buffer = pointers[index], .recv_buffer = const_cast<void *>(pointers[index]),
                .element_count = index < 2 ? static_cast<size_t>(input.seq_len) : 1u,
                .dtype = CollectiveDataType::INT32, .root_device_index = context.degree() - 1,
                .name = names[index]};
    }
    /** @brief Bind exactly the four committed banks, never the outcome/controller. */
    PipelineMTPMetadataCollective(ILocalTPContext &context, int participant, DeviceId device,
        PipelineForwardGraphEdges::PublicationBanks banks)
        : IComputeStage(device), context_(context), participant_(participant),
          field_count_(4), name_("pipeline_mtp_committed_state")
    {
        const std::array pointers{banks.restore_rows, banks.cached_tokens, banks.accepted_counts, banks.healthy};
        const std::array<const char *, 4> names{"mtp_restore_row", "mtp_cached_tokens",
            "mtp_accepted_count", "mtp_publication_ok"};
        for (size_t index = 0; index < field_count_; ++index)
            fields_[index] = {.kind = LocalTPCollectiveSidebandKind::Broadcast,
                .send_buffer = pointers[index], .recv_buffer = pointers[index],
                .element_count = 1,
                .dtype = CollectiveDataType::INT32, .root_device_index = context.degree() - 1,
                .name = names[index]};
    }
    /** @brief Record native broadcasts on the executor's exact non-null stream. */
    bool execute(IDeviceContext *context) override
    {
        return context && context->deviceId() == device() &&
            context_.collectiveSidebandSpanOnStream(std::span(fields_).first(field_count_),
                participant_, requireGPUStream(), name_.c_str());
    }
    /** @return Explicit collective ownership, never hidden inside a compute stage. */
    ComputeStageType type() const override { return ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE; }
    /** @return Stable semantic identity for the metadata producer boundary. */
    std::string name() const override { return name_; }
    /** @return Both supported native GPU collective backends have the same wire contract. */
    bool supportsBackend(ComputeBackendType backend) const override
    { return backend == ComputeBackendType::GPU_CUDA || backend == ComputeBackendType::GPU_ROCM; }
    /** @return These fixed native broadcasts belong inside retained captures. */
    bool isGraphCapturable() const override { return true; }
    /** @return The exact logical metadata payload, excluding unused bank capacity. */
    size_t estimatedMemoryBytes() const override
    {
        size_t elements = 0;
        for (size_t index = 0; index < field_count_; ++index)
            elements += fields_[index].element_count;
        return elements * sizeof(int32_t);
    }
    /** @return Bindings are persistent row owners, not new arena allocations. */
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    /** @return Immutable geometry only; diagnostics never read live row contents. */
    StageDumpInfo buildDumpInfoImpl() const override
    {
        StageDumpInfo info;
        info.addScalarInt("first_field_elements", static_cast<int>(fields_[0].element_count));
        info.addScalarInt("field_count", static_cast<int>(field_count_));
        info.addScalarInt("source_participant", context_.degree() - 1);
        return info;
    }
private:
    ILocalTPContext &context_;
    const int participant_;
    const size_t field_count_;
    const std::string name_;
    std::array<LocalTPCollectiveSidebandBuffer, 4> fields_{};
};

/** @brief One declared native send/receive, with no hidden host coordination. */
class PipelineActivationCollective final : public IComputeStage
{
public:
    /** @brief Borrow frozen native transport and an already-admitted local bank. */
    PipelineActivationCollective(DeviceId device, ILocalTPContext &context, int participant,
        TensorBase &hidden, size_t elements, CollectiveP2POpKind kind)
        : IComputeStage(device), context_(context), participant_(participant), hidden_(hidden),
          elements_(elements), kind_(kind) {}
    /** @brief Record or replay the exact P2P operation on the executor's stream.
     * @param context Execution context selected by the local graph executor.
     * @return Whether native transport accepted the operation. */
    bool execute(IDeviceContext *context) override
    {
        const auto execution = gpuExecution();
        if (!context || !execution.nativeStream() || !hidden_.gpu_data_ptr()) return false;
        const bool send = kind_ == CollectiveP2POpKind::Send;
        const CollectiveP2POp operation{.kind = kind_,
            .send_buffer = send ? hidden_.gpu_data_ptr() : nullptr,
            .recv_buffer = send ? nullptr : hidden_.gpu_data_ptr(),
            .count = elements_, .dtype = CollectiveDataType::FLOAT32,
            .peer = participant_ + (send ? 1 : -1)};
        return context_.groupedP2PRawOnStream({operation}, participant_, execution.nativeStream(), name().c_str());
    }
    /** @return Canonical collective classification for capture and scheduling. */
    ComputeStageType type() const override { return ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE; }
    /** @return Direction-qualified diagnostic identity, independent of ordinals. */
    std::string name() const override
    { return kind_ == CollectiveP2POpKind::Send ? "pipeline_activation_send" : "pipeline_activation_receive"; }
    /** @return True only for the native GPU backend admitted at construction. */
    bool supportsBackend(ComputeBackendType backend) const override
    { return backend == ComputeBackendType::GPU_CUDA || backend == ComputeBackendType::GPU_ROCM; }
    /** @return These fixed native edges require no host action during replay. */
    bool isGraphCapturable() const override { return true; }
    /** @return Exact read/send or receive/write payload, excluding unused arena rows. */
    size_t estimatedMemoryBytes() const override { return elements_ * sizeof(float); }
    /** @return Buffer direction lets the executor prepare the receive destination. */
    StageBufferRequirements getBufferRequirements() const override
    {
        StageBufferRequirements result;
        if (kind_ == CollectiveP2POpKind::Send)
            result.addInput("hidden", hidden_.shape(), BufferTensorType::FP32);
        else result.addOutput("hidden", hidden_.shape(), BufferTensorType::FP32);
        return result;
    }
    /** @return Both directions borrow the same participant-local persistent arena bank. */
    StageBufferContract bufferContract() const override
    {
        auto result = StageBufferContract::build();
        if (kind_ == CollectiveP2POpKind::Send) result.addInput(BufferId::HIDDEN_STATE);
        else result.addOutput(BufferId::HIDDEN_STATE);
        return result;
    }
    /** @return Tensor bindings for normal executor publication and diagnostics. */
    StageDumpInfo buildDumpInfoImpl() const override
    {
        StageDumpInfo result;
        if (kind_ == CollectiveP2POpKind::Send)
            result.addInput("hidden", &hidden_, hidden_.rows(), hidden_.cols());
        else result.addOutput("hidden", &hidden_, hidden_.rows(), hidden_.cols());
        return result;
    }
private:
    ILocalTPContext &context_;
    const int participant_;
    TensorBase &hidden_;
    const size_t elements_;
    const CollectiveP2POpKind kind_;
};
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
    DeviceGraphExecutor &executor, IDeviceContext &context, IWorkerGPUContext &gpu)
{
    if (!banks.valid() || context.deviceId() != device_)
        throw std::invalid_argument("Pipeline publication transport requires four distinct local metadata banks");
    if (publication_banks_)
    {
        if (*publication_banks_ != banks)
            throw std::logic_error("Frozen pipeline publication transport cannot rebind its metadata banks");
        auto existing = publication_transport_.lock();
        if (!existing || !publicationTransport(banks))
            throw std::logic_error("Retired pipeline publication transport cannot be resurrected");
        return existing;
    }
    // Only this setup wave records NCCL/RCCL work. Local acceptance policy
    // changes later compose this recording, never re-enter its rendezvous.
    auto transport = std::make_shared<PipelinePublicationTransport>();
    transport->banks = banks;
    transport->graph.addNode("pipeline_mtp_committed_state",
        std::make_unique<PipelineMTPMetadataCollective>(context_, participant_, device_, banks), device_);
    transport->graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
    if (!transport->cache.ensureCaptureStream(&gpu, device_, /*context_from_process_pool=*/true)) return {};
    transport->cache.perf_context = "pipeline_mtp_committed_state";
    DeviceGraphExecutor::DecodeCapturePolicy policy;
    policy.allow_cached_graph_replay = true;
    policy.collectives_graph_capturable = true;
    policy.defer_final_sync = true;
    policy.capture_boundary = [this](const std::string &boundary, void *stream) {
        return captureBoundary("mtp_publication_transport:" + boundary, stream);
    };
    if (!executor.executeDecodeWithCapturePolicy(transport->graph, &context, &transport->cache,
            transport->cache.capture_stream, &gpu, nullptr, policy, nullptr,
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)) return {};
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

PipelineForwardGraphEdges::PipelineForwardGraphEdges(ILocalTPContext &context, int participant,
    TensorBase &hidden, int width, std::optional<FollowerState> follower)
    : context_(context), participant_(participant), hidden_(hidden), width_(width),
      device_(participant >= 0 && participant < context.degree()
          ? context.devices().at(participant).toLocalDeviceId() : DeviceId::cpu()), follower_(follower)
{
    if (context.degree() < 2 || participant < 0 || participant >= context.degree() ||
        !device_.is_gpu() || width <= 0 || hidden.native_type() != TensorType::FP32 ||
        (context.backend() != CollectiveBackendType::NCCL && context.backend() != CollectiveBackendType::RCCL))
        throw std::invalid_argument("Pipeline forward requires a frozen native GPU context and local FP32 bank");
    if (follower_ && (!follower_->valid() || participant + 1 == context.degree()))
        throw std::invalid_argument("Pipeline follower checkpoint must be complete and belong to a nonterminal participant");
}

bool PipelineForwardGraphEdges::encloses(const ForwardInput &input) noexcept
{
    // Select from semantic ownership, never M or the request's global MTP flag.
    // Validation in append rejects contradictory fields instead of interpreting
    // an incomplete verifier as an ordinary graph with no activation receive.
    return input.device_prefill_chunk.has_value() ||
           input.execution_role == ForwardExecutionRole::GroupedMTPVerifier ||
           input.execution_role == ForwardExecutionRole::MTPCondition;
}

void PipelineForwardGraphEdges::validateInput(const ForwardInput &input) const
{
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
    const bool tail = participant_ + 1 == context_.degree();
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
        (!prefill && !verifier && !condition) ||
        (verifier && !tail && (!follower_ || input.kv_cache != follower_->checkpoint.cache)) ||
        input.external_hidden_state != (participant_ > 0 ? &hidden_ : nullptr) ||
        size_t(input.seq_len) > hidden_.numel() / size_t(width_))
        throw std::invalid_argument("Pipeline forward changed its exact participant, bank, semantic role or physical row geometry");
}

void PipelineForwardGraphEdges::validateFollowerMTPInput(const ForwardInput &input) const
{
    if (participant_ + 1 == context_.degree() ||
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
    const bool tail = participant_ + 1 == context_.degree();
    const auto roots = graph.getRootNodes();
    const auto leaves = graph.getLeafNodes();
    if (roots.empty() || leaves.empty()) throw std::invalid_argument("Pipeline forward model graph is empty");
    const size_t elements = size_t(input.seq_len) * size_t(width_);
    std::string preparation;
    if (verifier || condition)
    {
        // These collectives precede both activation receipt and local compute.
        // A follower cannot consume stale rows while waiting for an activation.
        preparation = "pipeline_mtp_rows";
        graph.addNode(preparation,
            std::make_unique<PipelineMTPMetadataCollective>(context_, participant_, input), device_);
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
    if (participant_ > 0)
    {
        constexpr const char *receive = "pipeline_activation_receive";
        graph.addNode(receive, std::make_unique<PipelineActivationCollective>(device_, context_,
            participant_, hidden_, elements, CollectiveP2POpKind::Recv), device_);
        if (!preparation.empty()) graph.addDependency(receive, preparation);
        for (const auto &root : roots) graph.addDependency(root, receive);
    }
    else if (!preparation.empty())
        for (const auto &root : roots) graph.addDependency(root, preparation);
    if (participant_ + 1 < context_.degree())
    {
        constexpr const char *send = "pipeline_activation_send";
        graph.addNode(send, std::make_unique<PipelineActivationCollective>(device_, context_,
            participant_, hidden_, elements, CollectiveP2POpKind::Send), device_);
        for (const auto &leaf : leaves) graph.addDependency(send, leaf);
        graph.setTerminalNode(send);
    }
}

void PipelineForwardGraphEdges::validatePublication(
    const MTPSpeculativeStatePublicationStage::Params &params) const
{
    using Publication = MTPSpeculativeStatePublicationStage;
    const bool tail = participant_ + 1 == context_.degree();
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
    const auto *transport = publicationTransport(PublicationBanks::from(params));
    if (!transport) return false;
    const bool tail = participant_ + 1 == context_.degree();
    const GPUOrderedTimelineStep commit{.name = "local accepted-state commit", .capture = &local};
    const GPUOrderedTimelineStep exchange{.name = "tail committed-state exchange", .capture = transport};
    const auto steps = tail ? std::array{commit, exchange} : std::array{exchange, commit};
    return destination.buildOrderedTimelineTransaction(steps);
}

bool PipelineForwardGraphEdges::captureBoundary(const std::string &boundary, void *stream) const
{
    return stream && context_.graphCaptureBoundaryOnStream("pipeline_forward:" + boundary,
        participant_, stream, collective_timeout_policy::effectiveCollectTimeoutMs(0));
}
}
