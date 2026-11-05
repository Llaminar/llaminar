/**
 * @file PooledImmutableNativeTransfer.cpp
 * @brief Native NCCL/RCCL endpoint binding for original ordered continuation PLE graphs.
 *
 * The original Rank-issued communicator and prepared workers retain execution
 * authority. Metadata stores immutable native operands and actual receive evidence;
 * no host code reads a count, interval, error or receipt during execution.
 */
#include "PooledImmutableNativeTransfer.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/BackendManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace llaminar2::qwen4exp {
void PooledNativeTransferSpan::require(DeviceId device, std::size_t alignment) const
{
    if (!device.is_gpu() || !alignment || offset_ % alignment || !bytes_)
        throw std::invalid_argument("Native pooled span requires an exact positive aligned GPU declaration");
    std::visit([&](const auto& owner) {
        if (!owner) throw std::invalid_argument("Native pooled span lost its original physical owner");
        using Owner = std::decay_t<decltype(owner)>;
        if constexpr (std::is_same_v<Owner, std::shared_ptr<TensorBase>>) {
            if ((role_ && *role_ >= BufferId::_COUNT) || owner->transferStorageOwner() != owner.get() || owner->is_view() ||
                offset_ > owner->size_bytes() || bytes_ > owner->size_bytes() - offset_ ||
                (owner->current_device() && owner->current_device() != device))
                throw std::invalid_argument("Native pooled span changed its original canonical tensor role/subrange");
        } else if (role_ || !owner->isBound() || owner->device() != device || !owner->contains(offset_, bytes_))
            throw std::invalid_argument("Native pooled span changed its original fixed transfer allocation");
    }, owner_);
}
void* PooledNativeTransferSpan::pointer(DeviceId device, std::size_t alignment) const
{
    require(device, alignment);
    return std::visit([&](const auto& owner) -> void* {
        using Owner = std::decay_t<decltype(owner)>;
        void* address;
        if constexpr (std::is_same_v<Owner, std::shared_ptr<TensorBase>>) {
            if (owner->current_device() != device || !owner->gpu_data_ptr())
                throw std::logic_error("Native pooled tensor has no original prepared GPU address");
            address = static_cast<std::byte*>(owner->gpu_data_ptr()) + offset_;
        } else address = owner->mutableDeviceData(offset_);
        if (reinterpret_cast<std::uintptr_t>(address) % alignment)
            throw std::logic_error("Native pooled span lost its prepared native alignment");
        return address;
    }, owner_);
}
PooledNativeTransferSpan PooledNativeTransferSpan::resized(std::size_t bytes) const
{
    if (!bytes) throw std::invalid_argument("Native pooled source bank must be positive");
    std::visit([&](const auto& owner) {
        if (!owner) throw std::invalid_argument("Native pooled source bank lost its original owner");
        using Owner = std::decay_t<decltype(owner)>;
        if constexpr (std::is_same_v<Owner, std::shared_ptr<TensorBase>>) {
            if (offset_ > owner->size_bytes() || bytes > owner->size_bytes() - offset_)
                throw std::out_of_range("Native pooled source bank exceeds its original tensor");
        } else if (!owner->contains(offset_, bytes))
            throw std::out_of_range("Native pooled source bank exceeds its original transfer allocation");
    }, owner_);
    auto result = *this; result.bytes_ = bytes; return result;
}
bool PooledNativeTransferSpan::sameOwner(const PooledNativeTransferSpan& other) const noexcept
{
    if (owner_.index() != other.owner_.index()) return false;
    return std::visit([&](const auto& owner) {
        const auto& right = std::get<std::decay_t<decltype(owner)>>(other.owner_);
        return owner.get() == right.get() && !owner.owner_before(right) && !right.owner_before(owner);
    }, owner_);
}
std::uint64_t PooledNativeTransferSpan::identity() const noexcept
{
    std::uint64_t value = 1469598103934665603ULL;
    const auto mix = [&](std::uint64_t word) { value = (value ^ word) * 1099511628211ULL; };
    mix(owner_.index());
    std::visit([&](const auto& owner) { mix(reinterpret_cast<std::uintptr_t>(owner.get())); }, owner_);
    mix(role_ ? static_cast<std::uint64_t>(*role_) : static_cast<std::uint64_t>(BufferId::_COUNT));
    mix(offset_); mix(bytes_);
    return value ? value : 1;
}

PooledImmutableNativeTransfer::PooledImmutableNativeTransfer(
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort, PreparedGPUContextUsePermit original,
    CapturedTransferEndpoint side, int producer, int consumer, int participant, CapturedTransferMessage message,
    PooledNativeTransferSpan payload)
    : cohort_(std::move(cohort)), original_(std::move(original)), side_(side), producer_(producer),
      consumer_(consumer), participant_(participant), message_(message), payload_(std::move(payload))
{
}

PooledImmutableNativeTransfer& PooledImmutableNativeTransfer::operator=(PooledImmutableNativeTransfer other) noexcept
{
    using std::swap;
    swap(cohort_, other.cohort_);
    swap(original_, other.original_);
    swap(side_, other.side_);
    swap(producer_, other.producer_);
    swap(consumer_, other.consumer_);
    swap(participant_, other.participant_);
    swap(message_, other.message_);
    swap(authority_, other.authority_);
    swap(payload_, other.payload_);
    swap(extent_, other.extent_);
    swap(slice_, other.slice_);
    swap(frame_, other.frame_);
    swap(nominal_, other.nominal_);
    swap(layout_, other.layout_);
    swap(operands_, other.operands_);
    swap(receipts_, other.receipts_);
    swap(prepared_, other.prepared_);
    swap(borrowed_, other.borrowed_);
    swap(source_bytes_, other.source_bytes_);
    swap(source_origin_, other.source_origin_);
    swap(slice_message_, other.slice_message_);
    swap(operand_published_, other.operand_published_);
    swap(sealed_, other.sealed_);
    swap(addresses_, other.addresses_);
    swap(backend_, other.backend_);
    swap(runtime_generation_, other.runtime_generation_);
    return *this;
}

PooledImmutableNativeTransfer PooledImmutableNativeTransfer::fixed(
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort, PreparedGPUContextUsePermit original,
    CapturedTransferEndpoint side, int producerCoordinate, int consumerCoordinate, CapturedTransferMessage message,
    PooledNativeTransferSpan payload)
{
    if (isGraphCaptureActive() || !cohort || !NativeByteEdge(producerCoordinate,consumerCoordinate).contains(cohort->degree(),
            side==CapturedTransferEndpoint::Producer?producerCoordinate:consumerCoordinate) || !message.key || !message.bytes ||
        (side != CapturedTransferEndpoint::Producer && side != CapturedTransferEndpoint::Consumer))
        throw std::invalid_argument("Native pooled transfer requires an original cold addressed fixed declaration");
    const auto participant = side == CapturedTransferEndpoint::Producer ? producerCoordinate : consumerCoordinate;
    cohort->requireConsumer(original.device(), participant);
    payload.require(original.device());
    if (message.bytes > payload.bytes())
        throw std::out_of_range("Native pooled message exceeds its original physical subrange");
    PooledImmutableNativeTransfer result(std::move(cohort), std::move(original), side,
        producerCoordinate, consumerCoordinate, participant, message, std::move(payload));
    result.authority_ = result.cohort_->context().nativeCompletionAuthority(participant);
    if (!result.authority_.valid())
        throw std::logic_error("Native pooled fixed endpoint lacks its original native communicator authority");
    result.requireCurrent();
    result.sealStorage();
    return result;
}

PooledImmutableNativeTransfer PooledImmutableNativeTransfer::withPrefix(
    PooledNativeCountedMessage nominal, PooledNativeCompletionLayout layout,
    std::shared_ptr<MappedHostTransferRegion> operandBacking, std::shared_ptr<DeviceTransferBuffer> receipts,
    PooledNativeTransferSpan extent, PooledNativeCompletionFrame frame) const
{
    requireCurrent();
    if (isGraphCaptureActive() || operand_published_ || extent_ || !operandBacking || !receipts ||
        layout.degree()!=static_cast<std::size_t>(cohort_->degree()) ||
        !layout.matchesStorageBytes(operandBacking->sizeBytes(), receipts->sizeBytes(),participant_) ||
        receipts->device() != device())
        throw std::invalid_argument("Counted native pooled binding requires original cold admitted completion roots");
    extent.require(device(), alignof(std::uint64_t));
    frame.interval.require(device(), alignof(std::uint64_t));
    frame.error.require(device(), alignof(std::int32_t));
    if (extent.bytes() != sizeof(std::uint64_t) || frame.interval.bytes() != 3 * sizeof(std::uint64_t) ||
        frame.error.bytes() != sizeof(std::int32_t) || frame.failureCode <= 0)
        throw std::invalid_argument("Native pooled completion requires exact original interval/error/count fields");
    (void)layout.operandOffset(nominal, participant_, NativeByteEdge(producer_,consumer_).peer(participant_));
    (void)layout.receiptOffset(nominal,participant_,NativeByteEdge(producer_,consumer_).peer(participant_));
    // deviceAlias checks the original closed mapping population. No metadata is
    // copied or device value observed while choosing the immutable binding.
    if (!operandBacking->deviceAlias(device()))
        throw std::invalid_argument("Native pooled operand is absent from its original participant mapping");
    auto result = *this;
    result.extent_ = std::move(extent);
    result.frame_ = std::move(frame);
    result.nominal_ = nominal;
    result.layout_ = layout;
    result.operands_ = std::move(operandBacking);
    result.receipts_ = std::move(receipts);
    result.sealStorage();
    return result;
}

PooledImmutableNativeTransfer PooledImmutableNativeTransfer::withSourceSlice(
    PooledNativeTransferSpan descriptor, std::size_t sourceBytes, std::uint64_t sourceOffsetOrigin) const
{
    requireCurrent();
    if (isGraphCaptureActive() || operand_published_ || !extent_ || slice_ ||
        side_ != CapturedTransferEndpoint::Producer || sourceBytes < message_.bytes ||
        sourceOffsetOrigin > std::numeric_limits<std::uint64_t>::max() - sourceBytes)
        throw std::invalid_argument("Native pooled source slice requires its original cold producer bank");
    descriptor.require(device(), alignof(NativeByteSourceSlice));
    if (descriptor.bytes() != sizeof(NativeByteSourceSlice) || !extent_->sameOwner(descriptor) ||
        extent_->offset() != descriptor.offset() + offsetof(NativeByteSourceSlice, live_bytes))
        throw std::invalid_argument("Native pooled source slice replaced its original live-byte publication");
    auto sourceBank = payload_.resized(sourceBytes);
    if (descriptor.sameOwner(sourceBank) && descriptor.offset() < sourceBank.offset() + sourceBank.bytes() &&
        sourceBank.offset() < descriptor.offset() + descriptor.bytes())
        throw std::invalid_argument("Native pooled slice descriptor overlaps its complete original source bank");
    auto result = *this;
    result.payload_ = std::move(sourceBank);
    result.slice_ = std::move(descriptor);
    result.source_bytes_ = sourceBytes;
    result.source_origin_ = sourceOffsetOrigin;
    result.slice_message_ = true;
    result.sealStorage();
    return result;
}

PooledImmutableNativeTransfer PooledImmutableNativeTransfer::withReceivedSourceSlice() const
{
    requireCurrent();
    if (isGraphCaptureActive() || operand_published_ || !extent_ || slice_message_ ||
        side_ != CapturedTransferEndpoint::Consumer)
        throw std::invalid_argument("Native pooled receive slice requires its original cold acquired extent");
    auto result = *this;
    result.slice_message_ = true;
    return result;
}

void* PooledImmutableNativeTransfer::pointer(const PooledNativeTransferSpan& span, std::size_t alignment) const
{
    return span.pointer(device(), alignment);
}

void PooledImmutableNativeTransfer::requireCurrent() const
{
    if (!cohort_ || !NativeByteEdge(producer_,consumer_).contains(cohort_->degree(),participant_))
        throw std::logic_error("Native pooled transfer lost its original addressed owner");
    cohort_->requireConsumer(device(), participant_);
    const auto current = GPUDeviceContextPool::instance().retainExistingPreparedContext(device(), original_.memory());
    original_.requireWorker(current.worker());
    if (current.generation() != original_.generation())
        throw std::logic_error("Native pooled transfer changed its original prepared generation");
    payload_.require(device());
    if (sealed_) {
        if (getBackendFor(device()) != backend_ || !runtime_generation_ ||
            backend_->deviceRuntimeGeneration(device().ordinal) != runtime_generation_)
            throw std::logic_error("Native pooled binding changed its original backend runtime generation");
        if (prepared_) prepared_->validate();
        const std::array<const PooledNativeTransferSpan*, 5> spans{&payload_,
            extent_ ? &*extent_ : nullptr, frame_ ? &frame_->interval : nullptr,
            frame_ ? &frame_->error : nullptr, slice_ ? &*slice_ : nullptr};
        for (std::size_t i = 0; i < spans.size(); ++i) if (spans[i]) {
            if (spans[i]->pointer(device()) != addresses_[i])
                throw std::logic_error("Native pooled binding lost its exact original prepared address");
            if (const auto tensor = spans[i]->tensor()) {
                if (!borrowed_[i])
                    throw std::logic_error("Native pooled binding lost its original allocation borrower");
                TransferEngine::instance().requireCapturedDeviceStorage(*borrowed_[i], *tensor,
                    device(), tensor->size_bytes());
            }
        }
    }
}

void PooledImmutableNativeTransfer::sealStorage()
{
    if (isGraphCaptureActive()) throw std::logic_error("Native pooled storage must be sealed before capture");
    const std::array<const PooledNativeTransferSpan*, 5> spans{&payload_,
        extent_ ? &*extent_ : nullptr, frame_ ? &frame_->interval : nullptr,
        frame_ ? &frame_->error : nullptr, slice_ ? &*slice_ : nullptr};
    std::array<TensorBase*, 5> roots{};
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>, 5> borrowed{};
    std::array<void*, 5> addresses{};
    std::size_t count = 0;
    for (std::size_t i = 0; i < spans.size(); ++i) if (spans[i]) {
        addresses[i] = spans[i]->pointer(device());
        if (const auto tensor = spans[i]->tensor()) {
            roots[count++] = tensor.get();
            for (std::size_t prior = 0; prior < i; ++prior)
                if (spans[prior] && spans[i]->sameOwner(*spans[prior])) { borrowed[i] = borrowed[prior]; break; }
            if (!borrowed[i]) borrowed[i] = TransferEngine::instance().retainCapturedDeviceStorage(
                *tensor, device(), tensor->size_bytes());
        }
    }
    std::optional<Qwen4ExpPreparedStorage> seal;
    if (count) {
        seal.emplace(device(), std::span<TensorBase* const>(roots.data(), count));
        seal->seal();
    }
    auto* backend = getBackendFor(device());
    const auto generation = backend ? backend->deviceRuntimeGeneration(device().ordinal) : 0;
    if (!generation) throw std::logic_error("Native pooled binding requires its original prepared runtime generation");
    // Acquire every new borrower before replacing its former sealed set.
    borrowed_ = std::move(borrowed); prepared_ = std::move(seal); addresses_ = addresses;
    backend_ = backend; runtime_generation_ = generation; sealed_ = true;
}

void PooledImmutableNativeTransfer::prepare(void* stream)
{
    requireCurrent();
    if (isGraphCaptureActive() || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
        !original_.worker()->ownsCurrentThread())
        throw std::invalid_argument("Native pooled preparation requires its exact original parent stream/worker");
    if (operand_published_) return;
    if (extent_) {
        if (!authority_.valid())
            throw std::logic_error("Native pooled binding has no original completed collective authority");
        auto* interval = static_cast<std::uint64_t*>(pointer(frame_->interval, alignof(std::uint64_t)));
        auto* error = static_cast<std::int32_t*>(pointer(frame_->error, alignof(std::int32_t)));
        auto* receipt = static_cast<ncclLlaminarByteCompletion*>(receipts_->mutableDeviceData(
            layout_->receiptOffset(*nominal_,participant_,NativeByteEdge(producer_,consumer_).peer(participant_))));
        const ncclLlaminarCompletionOperand operand{interval, interval + 1, interval + 2, error,
            receipt, nullptr, message_.key, frame_->failureCode, 0};
        auto* target = static_cast<std::byte*>(operands_->mutableHostData()) +
            layout_->operandOffset(*nominal_, participant_,NativeByteEdge(producer_,consumer_).peer(participant_));
        detail::sealPooledNativeOperand(*reinterpret_cast<ncclLlaminarCompletionOperand*>(target), operand);
    }
    operand_published_ = true;
}

void PooledImmutableNativeTransfer::enqueue(void* stream) const
{
    requireCurrent();
    if (!sealed_ || !operand_published_ || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
        !original_.worker()->ownsCurrentThread())
        throw std::logic_error("Native pooled enqueue lost its original sealed parent stream/worker");
    auto& context = cohort_->context();
    auto bytes = NativeCollectiveBytes::fixed(message_.bytes);
    if (extent_) {
        const auto* operand = reinterpret_cast<const ncclLlaminarCompletionOperand*>(
            static_cast<std::byte*>(operands_->deviceAlias(device())) +
            layout_->operandOffset(*nominal_, participant_,NativeByteEdge(producer_,consumer_).peer(participant_)));
        auto* extent = static_cast<std::uint64_t*>(pointer(*extent_, alignof(std::uint64_t)));
        if (slice_)
            bytes = NativeCollectiveBytes::sourceSlice(static_cast<NativeByteSourceSlice*>(
                pointer(*slice_, alignof(NativeByteSourceSlice))), source_bytes_, message_.bytes, operand, source_origin_);
        else if (slice_message_)
            bytes = NativeCollectiveBytes::receiveSlice(extent, message_.bytes, operand);
        else
            bytes = NativeCollectiveBytes::prefix(extent, message_.bytes, operand);
    }
    auto* payload = pointer(payload_);
    if (!context.nativeTransferBytesOnStream(payload, bytes, NativeByteEdge(producer_,consumer_), participant_, stream,
            "original_pooled_ple_native_transfer")) {
        // Cancellation is the existing communicator-wide failure authority.
        // The enclosing Rank must still join all original native parents.
        context.signalNativeCompletionAbort();
        throw std::runtime_error("Original pooled NCCL/RCCL transfer enqueue failed");
    }
}

NativeReceiveCompletionView PooledImmutableNativeTransfer::completion() const
{
    requireCurrent();
    if (!sealed_ || !extent_)
        throw std::logic_error("Fixed or unprepared native pooled transfer has no counted receive proof");
    return authority_.receive(
        static_cast<const ncclLlaminarByteCompletion*>(receipts_->deviceData(layout_->receiptOffset(*nominal_,participant_,NativeByteEdge(producer_,consumer_).peer(participant_)))));
}

std::uint64_t PooledImmutableNativeTransfer::captureIdentity() const
{
    requireCurrent();
    std::uint64_t value = 1469598103934665603ULL;
    const auto mix = [&](std::uint64_t word) { value = (value ^ word) * 1099511628211ULL; };
    mix(reinterpret_cast<std::uintptr_t>(&cohort_->context()));
    mix(original_.generation()); mix(static_cast<std::uint64_t>(device().type)); mix(device().ordinal);
    mix(producer_); mix(consumer_); mix(participant_); mix(message_.key); mix(message_.bytes);
    mix(reinterpret_cast<std::uintptr_t>(authority_.binding().abortFlag));
    mix(reinterpret_cast<std::uintptr_t>(authority_.binding().deviceCommunicator));
    const auto span = [&](const PooledNativeTransferSpan& item) { mix(item.identity()); };
    span(payload_);
    if (extent_) {
        span(*extent_); span(frame_->interval); span(frame_->error);
        mix(static_cast<std::uint64_t>(frame_->failureCode));
        mix(reinterpret_cast<std::uintptr_t>(operands_.get()));
        mix(reinterpret_cast<std::uintptr_t>(receipts_.get()));
        mix(layout_->operandOffset(*nominal_, participant_,NativeByteEdge(producer_,consumer_).peer(participant_)));
        mix(layout_->receiptOffset(*nominal_,participant_,NativeByteEdge(producer_,consumer_).peer(participant_)));
    }
    if (slice_) span(*slice_);
    mix(source_bytes_); mix(source_origin_); mix(slice_message_);
    if (sealed_) {
        mix(reinterpret_cast<std::uintptr_t>(backend_)); mix(runtime_generation_);
        for (const auto address : addresses_) mix(reinterpret_cast<std::uintptr_t>(address));
    }
    return value ? value : 1;
}
} // namespace llaminar2::qwen4exp
