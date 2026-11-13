/**
 * @file PipelineNativeByteChannel.cpp
 * @brief Exact PMA-backed native pipeline endpoint frames and public tensor publication.
 *
 * Frame preparation and validation execute on the original participant stream.
 * The native byte ABI transfers only useful prefixes and writes the original
 * receipts. Tensor outputs publish only after receipt/row validation; a source
 * retains its original read frontier throughout every addressed consumer edge.
 */
#include "PipelineNativeByteChannel.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/NativeContextPreparation.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>
#include <utility>
namespace llaminar2 {
namespace {
/** @brief A failed native submission must publish absorbing original abort before any unwinding. */
void abortNativePipeline(ILocalTPContext &context) noexcept {
    try { if (!context.signalNativeCompletionAbort()) std::terminate(); }
    catch (...) { std::terminate(); }
}
}
/** @brief Physical endpoint roots retire buffer storage before its exact allocation claim. */
struct PipelineNativeByteChannel::Endpoint final {
    PreparedGPUContextUsePermit original;
    NativeCompletionAuthority authority;
    IBackend *backend;
    void *stream;
    PhysicalMemoryAllocationLease claim;
    std::shared_ptr<DeviceTransferBuffer> frame;
    bool initialization_pending = false;
    /** @brief Retain only original prepared worker/native identities before materialization. */
    Endpoint(PreparedGPUContextUsePermit permit, NativeCompletionAuthority native, void *setup)
        : original(std::move(permit)), authority(std::move(native)), backend(getBackendFor(original.device())), stream(setup) {}
    /** @brief A partial cold failure still drains its actual original initialization before release. */
    void finishInitialization() {
        if (!initialization_pending) return;
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        for (;;) {
            GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
            original.worker()->submitAndWait([&] {
                state = original.worker()->queryStreamExecutionState(stream, "native pipeline byte initialization");
            });
            if (state == GPUStreamExecutionState::Complete) break;
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Original native pipeline initialization did not retire");
            std::this_thread::yield();
        }
        initialization_pending = false;
    }
    /** @brief Never free a partially submitted native initialization's original frame. */
    ~Endpoint() { try { finishInitialization(); } catch (...) { std::terminate(); } }
};
PipelineNativeByteChannel::PipelineNativeByteChannel(ILocalTPContext &context, int source,
    std::vector<int> consumers, std::size_t capacity)
    : original_(context.borrowOriginalNativeContext()), members_(context.devices()), source_(source),
      consumers_(std::move(consumers)), capacity_(capacity) {}
std::shared_ptr<PipelineNativeByteChannel> PipelineNativeByteChannel::create(ILocalTPContext &context, int source,
    std::span<const int> consumers, std::size_t capacity,
    const std::shared_ptr<PhysicalMemoryAuthority> &memory, std::span<void *const> streams)
{
    if (isGraphCaptureActive() || !memory || !capacity || capacity > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) || context.degree() < 2 ||
        context.devices().size() != static_cast<std::size_t>(context.degree()) ||
        source < 0 || source >= context.degree() || consumers.empty() || streams.size() != context.devices().size())
        throw std::invalid_argument("Native pipeline bytes require complete original cold membership and admission");
    const auto backend = context.devices().front().toLocalDeviceId().is_cuda()
        ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
    if (context.backend() != backend || !context.isLocal())
        throw std::invalid_argument("Native pipeline bytes require the original local NCCL/RCCL authority");
    for (std::size_t i = 0; i < consumers.size(); ++i) {
        if (consumers[i] < 0 || consumers[i] >= context.degree() || consumers[i] == source ||
            std::find(consumers.begin(), consumers.begin() + i, consumers[i]) != consumers.begin() + i)
            throw std::invalid_argument("Native pipeline bytes have a foreign or repeated actual consumer");
    }
    auto result = std::shared_ptr<PipelineNativeByteChannel>(new PipelineNativeByteChannel(
        context, source, {consumers.begin(), consumers.end()}, capacity));
    const auto native = result->original_.acquire();
    for (const int consumer : consumers) for (const int participant : {source, consumer}) {
        const auto device = result->device(participant);
        if (!device.is_gpu() || device.type != result->device(source).type ||
            reinterpret_cast<std::uintptr_t>(streams[participant]) <= 2)
            throw std::invalid_argument("Native pipeline endpoint changed its backend or original explicit stream");
        auto permit = GPUDeviceContextPool::instance().retainExistingPreparedContext(device, memory);
        if (permit.worker()->ownsCurrentThread())
            throw std::invalid_argument("Native pipeline byte setup cannot occupy an original participant worker");
        auto endpoint = std::make_shared<Endpoint>(std::move(permit),
            native.context().nativeCompletionAuthority(participant), streams[participant]);
        endpoint->claim = memory->claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, endpointBytes());
        endpoint->original.worker()->submitAndWait([&] {
            endpoint->frame = TransferEngine::instance().allocateDeviceTransferBuffer(endpointBytes(), device);
            const NativePipelineByteInvocation invocation{endpoint->frame->mutableDeviceData(), 1, capacity,
                nullptr, 0, endpoint->authority.binding()};
            endpoint->initialization_pending = true;
            if (!endpoint->backend || !endpoint->backend->enqueueNativePipelineByteFrame(invocation,
                    NativePipelineByteOperation::Initialize, device.ordinal, endpoint->stream))
                throw std::runtime_error("Native pipeline byte frame initialization failed");
        });
        endpoint->finishInitialization();
        PerfStatsCollector::addCounter("generation", "pipeline_native_byte_endpoint_materializations", 1.0,
            "graph_setup", device.toString(), {{"source", std::to_string(source)}, {"consumer", std::to_string(consumer)},
                {"participant", std::to_string(participant)}, {"transport", device.is_cuda() ? "nccl" : "rccl"},
                {"frame_bytes", std::to_string(endpointBytes())}, {"capacity", std::to_string(capacity)},
                {"mapped_payload_bytes", "0"}});
        result->endpoints_.emplace(std::pair{consumer, participant}, std::move(endpoint));
    }
    return result;
}
void PipelineNativeByteChannel::requireOriginal(ILocalTPContext &context, int participant) const {
    const auto original = original_.acquire();
    if (!original_.sameOriginal(context.borrowOriginalNativeContext()) || context.devices() != members_ ||
        participant < 0 || static_cast<std::size_t>(participant) >= members_.size())
        throw std::invalid_argument("Native pipeline port substituted its original communicator/member authority");
}
DeviceId PipelineNativeByteChannel::device(int participant) const {
    if (participant < 0 || static_cast<std::size_t>(participant) >= members_.size())
        throw std::out_of_range("Native pipeline byte endpoint is outside original membership");
    return members_[participant].toLocalDeviceId();
}
bool PipelineNativeByteChannel::isConsumer(int participant) const noexcept
{ return std::find(consumers_.begin(), consumers_.end(), participant) != consumers_.end(); }
std::pair<DeviceId, const void *> PipelineNativeByteChannel::terminalObservation(int consumer, int participant) const
{
    const auto &endpoint = *endpoints_.at({consumer, participant});
    return {device(participant), endpoint.frame->deviceData()};
}
void PipelineNativeByteChannel::enqueue(ILocalTPContext &native, int participant, void *payload, CapturedTransferMessage message,
    const std::int32_t *rows, std::uint64_t row_bytes, std::int32_t *received_rows,
    NativePipelineByteObservation observation, void *stream) const
{
    if (!payload || !message.fits(capacity_) || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
        (!isSource(participant) && !isConsumer(participant)) || (received_rows && !isConsumer(participant)))
        throw std::invalid_argument("Native pipeline byte replay lost its actual endpoint, payload or stream");
    if (!original_.sameOriginal(native.borrowOriginalNativeContext()) || native.devices() != members_)
        throw std::logic_error("Native pipeline communicator membership changed after capture binding");
    try {
    for (const int consumer : consumers_) {
        if (!isSource(participant) && participant != consumer) continue;
        const auto &endpoint = *endpoints_.at({consumer, participant});
        const auto id = device(participant);
        const auto current = GPUDeviceContextPool::instance().retainExistingPreparedContext(id, endpoint.original.memory());
        endpoint.original.requireWorker(current.worker());
        if (current.generation() != endpoint.original.generation())
            throw std::logic_error("Native pipeline endpoint changed original prepared generation");
        void *const memory = endpoint.frame->mutableDeviceData();
        const NativePipelineByteInvocation invocation{memory, message.key, message.bytes, rows, row_bytes,
            endpoint.authority.binding(), received_rows, observation};
        if (!invocation.valid() || !endpoint.backend->enqueueNativePipelineByteFrame(invocation,
                NativePipelineByteOperation::Prepare, id.ordinal, stream)) {
            throw std::runtime_error("Native pipeline byte preparation failed");
        }
        const NativeCountedAllGatherLayout layout(1);
        auto *extent = static_cast<std::uint64_t *>(endpoint.frame->mutableDeviceData(layout.extents()));
        const auto *operand = static_cast<const ncclLlaminarCompletionOperand *>(endpoint.frame->mutableDeviceData(layout.operands()));
        if (!native.nativeTransferBytesOnStream(payload, NativeCollectiveBytes::prefix(extent, message.bytes, operand),
                NativeByteEdge(source_, consumer), participant, stream, "pipeline_domain_native_bytes") ||
            !endpoint.backend->enqueueNativePipelineByteFrame(invocation, NativePipelineByteOperation::Validate, id.ordinal, stream)) {
            throw std::runtime_error("Native pipeline bytes did not publish complete original receipt evidence");
        }
    }
    } catch (...) {
        abortNativePipeline(native);
        throw;
    }
}

PipelineNativeByteBinding::PipelineNativeByteBinding(std::shared_ptr<PipelineNativeByteChannel> channel, int participant,
    CapturedTransferMessage message, std::shared_ptr<TensorBase> owner, std::size_t offset,
    std::optional<PipelineNativeRowCount> rows)
    : PipelineNativeByteBinding(std::move(channel), participant, message, std::move(owner), offset,
          rows ? Rows{std::move(*rows)} : Rows{}, CheckedRows{}) {}

PipelineNativeByteBinding::PipelineNativeByteBinding(std::shared_ptr<PipelineNativeByteChannel> channel, int participant,
    CapturedTransferMessage message, std::shared_ptr<TensorBase> owner, std::size_t offset,
    PipelineNativeReceivedRows received)
    : PipelineNativeByteBinding(std::move(channel), participant, message, std::move(owner), offset,
          Rows{std::move(received)}, CheckedRows{}) {}

PipelineNativeByteBinding::PipelineNativeByteBinding(std::shared_ptr<PipelineNativeByteChannel> channel, int participant,
    CapturedTransferMessage message, std::shared_ptr<TensorBase> owner, std::size_t offset, Rows rows, CheckedRows)
    : channel_(std::move(channel)), participant_(participant), message_(message), owner_(owner), offset_(offset),
      data_(nullptr), rows_(std::move(rows))
{
    if (!channel_ || !owner || (!channel_->isSource(participant) && !channel_->isConsumer(participant)) ||
        !message.fits(channel_->capacity()) || offset > owner->size_bytes() ||
        message.bytes > owner->size_bytes() - offset || owner->current_device() != channel_->device(participant))
        throw std::invalid_argument("Native pipeline binding requires its original resident tensor subrange");
    capture_ = TransferEngine::instance().retainCapturedDeviceStorage(*owner, channel_->device(participant_), offset + message.bytes);
    data_ = static_cast<std::byte *>(owner->gpu_data_ptr()) + offset;
    const auto retain_rows = [&](const auto &count) {
        if (!count.owner || count.owner->native_type() != TensorType::INT32 || !count.row_bytes ||
            count.owner->current_device() != channel_->device(participant_) || count.offset % alignof(std::int32_t) ||
            count.offset > count.owner->size_bytes() || sizeof(std::int32_t) > count.owner->size_bytes() - count.offset ||
            message.bytes % count.row_bytes || message.bytes / count.row_bytes > std::uint64_t{std::numeric_limits<std::int32_t>::max()})
            throw std::invalid_argument("Native pipeline rows require their exact disjoint original INT32 field");
        constexpr auto address_limit = std::numeric_limits<std::uintptr_t>::max();
        const auto count_base = reinterpret_cast<std::uintptr_t>(count.owner->gpu_data_ptr());
        const auto payload_start = reinterpret_cast<std::uintptr_t>(data_);
        if (!count_base || count_base > address_limit - (count.offset + sizeof(std::int32_t)) ||
            payload_start > address_limit - message.bytes)
            throw std::invalid_argument("Native pipeline row or payload device span overflows");
        const auto count_start = count_base + count.offset;
        if (count_start % alignof(std::int32_t) ||
            (count_start < payload_start + message.bytes && payload_start < count_start + sizeof(std::int32_t)))
            throw std::invalid_argument("Native pipeline row field is misaligned or overlaps its actual payload device span");
        row_capture_ = TransferEngine::instance().retainCapturedDeviceStorage(*count.owner,
            channel_->device(participant_), count.offset + sizeof(std::int32_t));
        row_address_ = reinterpret_cast<std::int32_t *>(count_start);
    };
    if (const auto *count = std::get_if<PipelineNativeRowCount>(&rows_)) retain_rows(*count);
    if (const auto *received = std::get_if<PipelineNativeReceivedRows>(&rows_)) {
        if (!channel_->isConsumer(participant_))
            throw std::invalid_argument("Only the original native consumer may publish received rows");
        retain_rows(*received);
    }
}
PipelineNativeByteBinding::PipelineNativeByteBinding(std::shared_ptr<PipelineNativeByteChannel> channel, int participant,
    CapturedTransferMessage message, std::shared_ptr<const WorkspaceBufferLease> owner, std::size_t offset)
    : channel_(std::move(channel)), participant_(participant), message_(message), owner_(owner), offset_(offset), data_(nullptr)
{
    if (!channel_ || !owner || (!channel_->isSource(participant) && !channel_->isConsumer(participant)) ||
        !message.fits(channel_->capacity()) || owner->device() != channel_->device(participant) ||
        !owner->contains(offset, message.bytes))
        throw std::invalid_argument("Native pipeline binding requires its original workspace subrange");
    data_ = owner->data(offset);
}
PipelineNativeByteBinding &PipelineNativeByteBinding::operator=(PipelineNativeByteBinding other) noexcept {
    using std::swap;
    swap(channel_, other.channel_); swap(participant_, other.participant_); swap(message_, other.message_);
    swap(owner_, other.owner_); swap(offset_, other.offset_); swap(data_, other.data_);
    swap(rows_, other.rows_); swap(receipt_, other.receipt_); swap(row_address_, other.row_address_);
    swap(row_capture_, other.row_capture_); swap(capture_, other.capture_);
    return *this;
}
void *PipelineNativeByteBinding::data() const {
    if (const auto *tensor = std::get_if<std::shared_ptr<TensorBase>>(&owner_)) {
        if (!capture_) throw std::logic_error("Native pipeline tensor binding lost its original capture lease");
        TransferEngine::instance().requireCapturedDeviceStorage(*capture_, **tensor, channel_->device(participant_), offset_ + message_.bytes);
        if (static_cast<std::byte *>((*tensor)->gpu_data_ptr()) + offset_ != data_)
            throw std::logic_error("Native pipeline tensor address changed after capture");
    } else {
        const auto &workspace = std::get<std::shared_ptr<const WorkspaceBufferLease>>(owner_);
        if (workspace->device() != channel_->device(participant_) || !workspace->contains(offset_, message_.bytes) ||
            workspace->data(offset_) != data_) throw std::logic_error("Native pipeline workspace changed after capture");
    }
    return data_;
}
PipelineNativeByteBinding PipelineNativeByteBinding::withTerminalReceipt(DeviceGenerationCollectiveReceipt receipt) && {
    if (receipt_ || !receipt.isFixedHeadPipelineEndpoint() || receipt.bankBytes() != message_.bytes ||
        channel_->consumers_.size() != 1 || std::holds_alternative<std::monostate>(rows_))
        throw std::invalid_argument("Native terminal receipt requires one original counted addressed endpoint");
    receipt.requireFor(channel_->device(participant_));
    receipt_ = receipt;
    return std::move(*this);
}
void PipelineNativeByteBinding::enqueue(void *stream) const {
    // Revocation is checked before even coherence/event submission. The scoped
    // borrow prevents native communicator teardown throughout this host enqueue.
    const auto original = channel_->original_.acquire();
    void *const payload = data();
    const bool source = channel_->isSource(participant_);
    const auto *tensor = std::get_if<std::shared_ptr<TensorBase>>(&owner_);
    if (tensor) {
        if (source) TransferEngine::requireDeviceInput(tensor->get(), channel_->device(participant_), stream);
        else TransferEngine::requireDeviceOutput(tensor->get(), channel_->device(participant_), stream);
    }
    const auto row_address = [&](const auto &count) {
        if (!row_capture_ || !row_address_) throw std::logic_error("Native pipeline count lost its original capture borrow");
        TransferEngine::instance().requireCapturedDeviceStorage(*row_capture_, *count.owner, channel_->device(participant_),
            count.offset + sizeof(std::int32_t));
        const auto *address = static_cast<const std::byte *>(count.owner->gpu_data_ptr()) + count.offset;
        if (address != reinterpret_cast<const std::byte *>(row_address_))
            throw std::logic_error("Native pipeline count address changed after its capture binding");
        return row_address_;
    };
    const std::int32_t *rows = nullptr;
    std::int32_t *received_rows = nullptr;
    std::uint64_t row_bytes = 0;
    const auto device = channel_->device(participant_);
    const auto *received = std::get_if<PipelineNativeReceivedRows>(&rows_);
    if (const auto *count = std::get_if<PipelineNativeRowCount>(&rows_)) {
        rows = row_address(*count);
        TransferEngine::requireDeviceInput(count->owner.get(), device, stream);
        row_bytes = count->row_bytes;
    } else if (received) {
        received_rows = row_address(*received);
        TransferEngine::requireDeviceOutput(received->owner.get(), device, stream);
        row_bytes = received->row_bytes;
    }
    NativePipelineByteObservation observation;
    if (receipt_) {
        observation = {source ? receipt_->native(device) : nullptr, receipt_->pipelineEndpointCompletion(device)};
        TransferEngine::requireDeviceInput(receipt_->owner(), device, stream);
        TransferEngine::requireDeviceOutput(receipt_->owner(), device, stream);
    }
    channel_->enqueue(original.context(), participant_, payload, message_, rows, row_bytes, received_rows, observation, stream);
    if (receipt_) TransferEngine::publishDeviceWrite(receipt_->owner(), device, stream);
    if (tensor && !source) TransferEngine::publishDeviceWrite(tensor->get(), device, stream);
    if (received && (!tensor || (*tensor)->transferStorageOwner() != received->owner->transferStorageOwner()))
        TransferEngine::publishDeviceWrite(received->owner.get(), device, stream);
}
std::uint64_t PipelineNativeByteBinding::captureIdentity() const noexcept {
    auto value = reinterpret_cast<std::uintptr_t>(channel_.get());
    const auto mix = [&](std::uint64_t word) { value = (value ^ word) * 1099511628211ULL; };
    mix(participant_); mix(message_.key); mix(message_.bytes); mix(reinterpret_cast<std::uintptr_t>(data_)); mix(offset_);
    const auto mix_rows = [&](const auto &count) {
        mix(reinterpret_cast<std::uintptr_t>(count.owner.get())); mix(count.offset); mix(count.row_bytes);
        mix(reinterpret_cast<std::uintptr_t>(row_address_));
    };
    mix(rows_.index());
    if (receipt_) mix(receipt_->captureIdentity());
    if (const auto *count = std::get_if<PipelineNativeRowCount>(&rows_)) mix_rows(*count);
    if (const auto *received = std::get_if<PipelineNativeReceivedRows>(&rows_)) mix_rows(*received);
    return value ? value : 1;
}
} // namespace llaminar2
