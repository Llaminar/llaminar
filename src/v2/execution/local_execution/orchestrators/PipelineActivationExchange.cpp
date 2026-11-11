/**
 * @file PipelineActivationExchange.cpp
 * @brief Capture-local native and heterogeneous pipeline activation boundaries.
 *
 * The leader's channel receive precedes its native broadcast on the same
 * stream. Other TP members enqueue only that broadcast, so NCCL/RCCL retains
 * intra-domain P2P and all participants consume their own arena storage.
 * TransferEngine owns channel ordering and tensor publication throughout.
 */
#include "PipelineActivationExchange.h"
#include "PipelineActivationFanout.h"
#include "planning/PipelineTransferMemory.h"
#include "backends/BackendManager.h"
#include "backends/NativeContextPreparation.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include <chrono>
#include <array>
#include <thread>
#include <new>
#include "memory/StageBufferContract.h"
#include "tensors/TensorClasses.h"
#include <stdexcept>

namespace llaminar2
{
namespace
{
/** @brief Stable native operation identity; allocation/scope/producer epoch disambiguate every replay. */
constexpr std::uint64_t kPipelineActivationNativeMessage = 0x41435446414e4f55ULL;
}

/** @brief Actual physical roots, declared so aliases/buffers retire before claims and original worker. */
struct PipelineActivationFanout::Storage final
{
    GPUContextProvenance original;
    const PipelineNativeDomain membership;
    ILocalTPContext &context;
    const int participant;
    const std::shared_ptr<CapturedTransferChannel> channel;
    const std::shared_ptr<PhysicalMemoryAuthority> memory;
    const std::uint64_t allocation, scope, native_message;
    const CapturedTransferPublication *original_publication = nullptr; ///< Leader-local alias, never a serialized peer pointer.
    NativeReceiveCompletionView completion = NativeReceiveCompletionView::notRequired();
    TransferEngine &transfer; ///< Same exact transfer authority for persistent setup and replay.
    IBackend *const backend;
    void *const setup_stream;
    PhysicalMemoryAllocationLease host_claim, device_claim;
    std::shared_ptr<MappedHostTransferRegion> operands;
    std::shared_ptr<DeviceTransferBuffer> device;
    bool initialization_pending = false;

    /** @brief Bind existing identities only; claims and native operations follow full validation. */
    Storage(DeviceId id, ILocalTPContext &ctx, int member, std::shared_ptr<CapturedTransferChannel> incoming,
        std::shared_ptr<PhysicalMemoryAuthority> authority, void *stream, TransferEngine &transfers, IBackend *resolved)
        : original(GPUDeviceContextPool::instance().retainExistingContextProvenance(id)),
          membership(id, &ctx, member), context(ctx), participant(member), channel(std::move(incoming)),
          memory(std::move(authority)), allocation(channel->identity().nonce),
          scope(channel->rankPlan() ? channel->rankPlan()->originalId().generation : allocation),
          native_message(channel->rankPlan() ? channel->rankPlan()->originalId().boundary : kPipelineActivationNativeMessage),
          transfer(transfers), backend(resolved), setup_stream(stream) {}

    /** @brief Observe only the exact cold initialization stream; no hot-path synchronization. */
    void completeInitialization()
    {
        if (!initialization_pending) return;
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        for (;;)
        {
            GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
            original.worker()->submitAndWait([&] {
                state = original.worker()->queryStreamExecutionState(setup_stream, "pipeline fanout original cold initialization");
            });
            if (state == GPUStreamExecutionState::Complete) break;
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Pipeline fanout exact initialization stream exceeded canonical deadline");
            std::this_thread::yield();
        }
        initialization_pending = false;
    }
    /** @brief Partial cold failure still proves native retirement before releasing actual allocations. */
    ~Storage() { try { completeInitialization(); } catch (...) { std::terminate(); } }
};

PipelineActivationFanout::PipelineActivationFanout(std::unique_ptr<Storage> storage)
    : storage_(std::move(storage)) {}
PipelineActivationFanout::~PipelineActivationFanout() = default;
size_t PipelineActivationFanout::hostBytes()
{ return PipelineTransferMemory::fanoutHostBytes(); }

std::shared_ptr<PipelineActivationFanout> PipelineActivationFanout::create(DeviceId device,
    ILocalTPContext &context, int participant, std::shared_ptr<CapturedTransferChannel> channel,
    const std::shared_ptr<PhysicalMemoryAuthority> &memory, void *setup_stream, TransferEngine &transfer)
{
    if (isGraphCaptureActive() || !channel || !memory || !device.is_gpu() ||
        reinterpret_cast<std::uintptr_t>(setup_stream) <= 2)
        throw std::invalid_argument("Pipeline fanout requires original admitted cold GPU ownership and exact stream");
    PipelineNativeDomain membership(device, &context, participant);
    if (channel->endpointDevice(CapturedTransferEndpoint::Consumer) != membership.leader())
        throw std::invalid_argument("Pipeline fanout changed the original receiving domain leader");
    auto storage = std::make_unique<Storage>(device, context, participant, std::move(channel), memory, setup_stream, transfer, transfer.resolveBackend(device));
    if (!storage->backend || storage->original.worker()->ownsCurrentThread())
        throw std::invalid_argument("Pipeline fanout cold creation requires the original off-worker setup owner");
    // Bind the same original prepared birth/PMA, not merely a numerically
    // identical device ordinal under another independent authority.
    const auto prepared = GPUDeviceContextPool::instance().retainExistingPreparedContext(device, memory);
    storage->original.requireWorker(prepared.worker());
    if (participant == 0)
    {
        storage->channel->requireOriginalEndpoint(CapturedTransferEndpoint::Consumer);
        auto *control = static_cast<CapturedTransferChannelControl *>(storage->channel->mapped_->deviceAlias(device));
        storage->original_publication = &control->consumer;
    }
    storage->original.worker()->submitAndWait([&] {
        (void)storage->original.worker()->queryStreamExecutionState(setup_stream, "pipeline fanout original setup stream");
    });
    const auto authority = context.nativeCompletionAuthority(participant);
    if (!authority.valid()) throw std::invalid_argument("Pipeline fanout lacks its original native completion authority");
    constexpr auto owner = PhysicalMemoryOwner::ActivationTransportStaging;
    storage->host_claim = memory->claimNewAllocation(DeviceId::cpu(), owner, hostBytes());
    storage->device_claim = memory->claimNewAllocation(device, owner, deviceBytes());
    const std::array devices{device};
    storage->operands = transfer.allocateMappedHostRegion(sizeof(PipelineActivationFanoutOperand), devices);
    storage->device = transfer.allocateDeviceTransferBuffer(deviceBytes(), device);
    auto *state = static_cast<PipelineActivationFanoutDeviceState *>(storage->device->mutableDeviceData());
    auto *operand = ::new (storage->operands->mutableHostData()) PipelineActivationFanoutOperand{};
    operand->allocation = storage->allocation;
    operand->scope = storage->scope;
    auto *alias = static_cast<PipelineActivationFanoutOperand *>(storage->operands->deviceAlias(device));
    storage->completion = authority.receive(&state->receipt);
    operand->native = {&alias->allocation, &alias->scope, &state->header.epoch, &state->first_error,
        &state->receipt, nullptr, storage->native_message, 1, 0};
    // Native metadata is immutable from this point. Only GPU-written header,
    // receipt and sticky error change, under original same-stream execution.
    storage->initialization_pending = true;
    storage->original.worker()->submitAndWait([&] {
        std::uint64_t timeout = 0;
        if (!storage->backend->prepareCapturedTransferChannelKernels(device.ordinal,
                collective_timeout_policy::kDefaultCollectiveTimeoutMs, &timeout) || !timeout ||
            !storage->backend->memset(state, 0, deviceBytes(), device.ordinal, setup_stream))
            throw std::runtime_error("Pipeline fanout original metadata initialization failed");
    });
    storage->completeInitialization();
    return std::shared_ptr<PipelineActivationFanout>(new PipelineActivationFanout(std::move(storage)));
}

void PipelineActivationFanout::requireOriginal(DeviceId device, ILocalTPContext *context,
    int participant, const std::shared_ptr<CapturedTransferChannel> &channel) const
{
    const auto &original = *storage_;
    original.membership.validate();
    if (device != original.original.device() || context != &original.context || participant != original.participant ||
        !channel || channel.get() != original.channel.get() || channel.owner_before(original.channel) ||
        original.channel.owner_before(channel) || channel->identity().nonce != original.allocation)
        throw std::invalid_argument("Pipeline activation fanout substituted its original endpoint or native member");
}

bool PipelineActivationFanout::enqueue(TensorBase &hidden, TensorBase &rows, size_t row_offset,
    std::uint64_t row_bytes, CapturedTransferMessage message,
    CapturedTransferReceiveCompletion leader_completion, void *stream) const
{
    auto &original = *storage_;
    requireOriginal(original.original.device(), &original.context, original.participant, original.channel);
    const auto device = original.original.device();
    if (!original.original.worker()->ownsCurrentThread() || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
        rows.native_type() != TensorType::INT32 || row_offset % alignof(std::int32_t) ||
        row_offset > rows.size_bytes() || sizeof(std::int32_t) > rows.size_bytes() - row_offset ||
        !row_bytes || !message.fits(original.channel->capacityBytes()) || message.bytes % row_bytes ||
        message.bytes > hidden.size_bytes() || (original.participant == 0) != leader_completion.valid() ||
        (original.participant == 0 && leader_completion.publication() != original.original_publication))
        throw std::invalid_argument("Pipeline native fanout has incomplete original live-row/stream/leader binding");
    auto &transfer = original.transfer;
    transfer.requireDeviceInput(&rows, device, stream);
    transfer.requireDeviceOutput(&hidden, device, stream);
    auto *state = static_cast<PipelineActivationFanoutDeviceState *>(original.device->mutableDeviceData());
    const void *source = original.participant == 0 ? static_cast<const void *>(leader_completion.publication()) : &state->header;
    const auto *operand = &static_cast<const PipelineActivationFanoutOperand *>(original.operands->deviceAlias(device))->native;
    // All fallible immutable geometry checks precede either native enqueue.
    const auto header = NativeCollectiveBytes::fixed(sizeof(PipelineActivationFanoutHeader));
    const auto payload = NativeCollectiveBytes::prefix(&state->header.message.bytes, message.bytes, operand);
    if (!original.context.nativeBroadcastBytesOnStream(source, &state->header, header, 0, original.participant,
            stream, "pipeline_activation_original_publication")) return false;
    if (!original.context.nativeBroadcastBytesOnStream(hidden.gpu_data_ptr(), hidden.gpu_data_ptr(), payload,
            0, original.participant, stream, "pipeline_activation_live_payload")) return false;
    const auto *local_rows = reinterpret_cast<const std::int32_t *>(static_cast<const std::byte *>(rows.gpu_data_ptr()) + row_offset);
    transfer.enqueuePipelineActivationFanoutValidation({state, local_rows, row_bytes, message.bytes, message.key,
        original.allocation, original.scope, original.native_message, original.completion, leader_completion}, device, stream);
    transfer.publishDeviceWrite(&hidden, device, stream);
    return true;
}

PipelineActivationExchange::PipelineActivationExchange(DeviceId device, TensorBase &hidden,
    size_t elements, NativePeer peer)
    : IComputeStage(device), hidden_(hidden), elements_(elements), transport_(peer),
      membership_(device, &peer.context, peer.participant)
{
    if (hidden.native_type() != TensorType::FP32 || !elements || elements > hidden.numel() ||
        (peer.direction != CollectiveP2POpKind::Send && peer.direction != CollectiveP2POpKind::Recv) ||
        (peer.direction == CollectiveP2POpKind::Send ? peer.participant + 1 >= peer.context.degree() : peer.participant == 0))
        throw std::invalid_argument("Pipeline native activation has invalid geometry or adjacent peer");
}

PipelineActivationExchange::PipelineActivationExchange(DeviceId device, std::shared_ptr<TensorBase> hidden,
    CapturedDomain domain, TransferEngine &transfer)
    : IComputeStage(device), hidden_(*(hidden ? hidden.get() : throw std::invalid_argument("Missing local activation owner"))),
      elements_(domain.message.bytes / sizeof(float)),
      transport_(DomainBinding{domain, std::move(hidden), std::nullopt, &transfer}),
      membership_(device, domain.context, domain.participant)
{
    auto &binding = std::get<DomainBinding>(transport_);
    const auto &port = binding.domain;
    if (!device.is_gpu() || bool(port.channel) == bool(port.native) ||
        !port.message.fits(port.native ? port.native->capacity() : port.channel->capacityBytes()) ||
        port.message.bytes % sizeof(float) || hidden_.native_type() != TensorType::FP32 || elements_ > hidden_.numel())
        throw std::invalid_argument("Pipeline domain activation requires an exact FP32 message and local bank");
    if (port.endpoint != CapturedTransferEndpoint::Producer && port.endpoint != CapturedTransferEndpoint::Consumer)
        throw std::invalid_argument("Pipeline domain activation has an invalid endpoint role");
    if (port.endpoint == CapturedTransferEndpoint::Producer && port.participant != 0)
        throw std::invalid_argument("Only the pipeline source domain leader may publish an activation");
    const auto leader = membership_.leader();
    if (port.native ? (port.native->device(port.native_participant) != device ||
            (port.endpoint == CapturedTransferEndpoint::Producer ? !port.native->isSource(port.native_participant)
                                                               : !port.native->isConsumer(port.native_participant)))
        : port.channel->endpointDevice(port.endpoint) != leader)
        throw std::invalid_argument("Pipeline channel must connect exact original domain leaders");
    if (port.fanout)
    {
        if (!port.context || port.endpoint != CapturedTransferEndpoint::Consumer)
            throw std::invalid_argument("Pipeline native fanout belongs only to a receiving native TP member");
        port.fanout->requireOriginal(device, port.context, port.participant, port.channel);
    }
    if (port.rows)
    {
        const auto &rows = *port.rows;
        if (!rows.owner || rows.owner->transferStorageOwner() != rows.owner.get() ||
            rows.owner->native_type() != TensorType::INT32 || !rows.row_bytes ||
            rows.row_bytes % sizeof(float) || port.message.bytes % rows.row_bytes ||
            rows.offset % alignof(std::int32_t) || rows.offset > rows.owner->size_bytes() ||
            sizeof(std::int32_t) > rows.owner->size_bytes() - rows.offset ||
            (port.context && port.endpoint == CapturedTransferEndpoint::Consumer && !port.native && !port.fanout))
            throw std::invalid_argument("Pipeline live rows require their original INT32 input and admitted native fanout");
    }
    if (port.native)
        binding.native_endpoint.emplace(port.native, port.native_participant, port.message, binding.owner, 0,
            port.rows ? std::optional{PipelineNativeRowCount{port.rows->owner, port.rows->offset, port.rows->row_bytes}} : std::nullopt);
    else if (port.participant == 0)
    {
        binding.endpoint = transfer.bindCapturedTransfer(port.channel, port.endpoint, port.message, binding.owner);
        if (port.rows)
            binding.endpoint = transfer.bindDeviceRowCountedTransfer(std::move(*binding.endpoint),
                port.rows->owner, port.rows->offset, port.rows->row_bytes);
    }
}

void PipelineActivationExchange::validateMembership() const
{ membership_.validate(); }

bool PipelineActivationExchange::sends() const noexcept
{
    if (const auto *native = std::get_if<NativePeer>(&transport_)) return native->direction == CollectiveP2POpKind::Send;
    return std::get<DomainBinding>(transport_).domain.endpoint == CapturedTransferEndpoint::Producer;
}

bool PipelineActivationExchange::execute(IDeviceContext *context)
{
    if (!context || context->deviceId() != device()) return false;
    void *stream = requireGPUStream();
    validateMembership();
    if (!hidden_.gpu_data_ptr()) return false;
    if (const auto *native = std::get_if<NativePeer>(&transport_))
    {
        const CollectiveP2POp operation{.kind = native->direction,
            .send_buffer = sends() ? hidden_.gpu_data_ptr() : nullptr,
            .recv_buffer = sends() ? nullptr : hidden_.gpu_data_ptr(),
            .count = elements_, .dtype = CollectiveDataType::FLOAT32,
            .peer = native->participant + (sends() ? 1 : -1)};
        return native->context.groupedP2PRawOnStream({operation}, native->participant, stream, name());
    }
    const auto &binding = std::get<DomainBinding>(transport_);
    const auto &port = binding.domain;
    // Only the domain leader touches the inter-vendor slot. Its acquire/copy/
    // publication precedes the native broadcast by exact stream order.
    if (binding.native_endpoint) {
        binding.native_endpoint->enqueue(stream);
        return true; // Every actual destination member received its own completed native prefix.
    }
    if (binding.endpoint) binding.transfer->enqueueCapturedTransfer(*binding.endpoint, stream);
    if (!sends() && port.context)
    {
        if (port.rows)
        {
            const auto receipt = binding.endpoint
                ? binding.transfer->capturedReceiveCompletion(*binding.endpoint, stream)
                : CapturedTransferReceiveCompletion::notRequired();
            return port.fanout->enqueue(hidden_, *port.rows->owner, port.rows->offset,
                port.rows->row_bytes, port.message, receipt, stream);
        }
        // A known fixed scalar/verifier extent contains only useful rows.
        TransferEngine::requireDeviceOutput(&hidden_, device(), stream);
        if (!port.context->broadcastRawOnStream(hidden_.gpu_data_ptr(), hidden_.gpu_data_ptr(), elements_,
                CollectiveDataType::FLOAT32, 0, port.participant, stream, name())) return false;
        TransferEngine::publishDeviceWrite(&hidden_, device(), stream);
    }
    return true;
}

std::string PipelineActivationExchange::name() const
{ return sends() ? "pipeline_activation_send" : "pipeline_activation_receive"; }

bool PipelineActivationExchange::supportsBackend(ComputeBackendType backend) const
{
    return (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
        (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM);
}

StageBufferRequirements PipelineActivationExchange::getBufferRequirements() const
{
    StageBufferRequirements result;
    if (sends()) result.addInput("hidden", hidden_.shape(), BufferTensorType::FP32);
    else result.addOutput("hidden", hidden_.shape(), BufferTensorType::FP32);
    return result;
}

StageBufferContract PipelineActivationExchange::bufferContract() const
{
    auto result = StageBufferContract::build();
    if (sends()) result.addInput(BufferId::HIDDEN_STATE); else result.addOutput(BufferId::HIDDEN_STATE);
    if (const auto *binding = std::get_if<DomainBinding>(&transport_); binding && binding->domain.rows)
        result.addInput(BufferId::PREFILL_CHUNK_GEOMETRY);
    return result;
}

StageDumpInfo PipelineActivationExchange::buildDumpInfoImpl() const
{
    StageDumpInfo result;
    if (sends()) result.addInput("hidden", &hidden_, hidden_.rows(), hidden_.cols());
    else result.addOutput("hidden", &hidden_, hidden_.rows(), hidden_.cols());
    return result;
}
}
