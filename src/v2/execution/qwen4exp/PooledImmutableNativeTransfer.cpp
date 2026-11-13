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
namespace {
/** @brief Immutable alignment of payload, count, identity, error and optional source slice. */
constexpr std::array<std::size_t, 5> spanAlignments{
    1, alignof(std::uint64_t), alignof(std::uint64_t), alignof(std::int32_t), alignof(NativeByteSourceSlice)};

/** @brief Shared byte/alias law for the existing native three-pointer completion ABI.
 * @param device Exact original GPU declaration, never a native readiness grant.
 * @param extent Original UInt64 live-byte publication.
 * @param identity Three contiguous original UInt64 identity words.
 * @param error Original sticky Int32 failure field.
 * @param payload Original bounded useful byte bank, disjoint from metadata.
 * @param failureCode Original positive failure code recorded in the immutable operand.
 */
void requireCompletionSpans(DeviceId device, const PooledNativeTransferSpan& extent,
    const PooledNativeTransferSpan& identity, const PooledNativeTransferSpan& error,
    const PooledNativeTransferSpan& payload, std::int32_t failureCode)
{
    payload.require(device);
    extent.require(device, alignof(std::uint64_t));
    identity.require(device, alignof(std::uint64_t));
    error.require(device, alignof(std::int32_t));
    if (extent.bytes() != sizeof(std::uint64_t) || identity.bytes() != 3 * sizeof(std::uint64_t) ||
        error.bytes() != sizeof(std::int32_t) || failureCode <= 0)
        throw std::invalid_argument("Native pooled completion requires exact original identity/error/count fields");
    const std::array<const PooledNativeTransferSpan*, 4> spans{&payload, &extent, &identity, &error};
    for (std::size_t i = 0; i < spans.size(); ++i)
        for (std::size_t prior = 0; prior < i; ++prior)
            spans[i]->requireDisjoint(*spans[prior]);
}
}

void PooledNativeCompletionFrame::require(PooledNativeCountedMessage nominal, DeviceId device,
    const PooledNativeTransferSpan& extent, const PooledNativeTransferSpan& payload) const
{
    if (nominal == PooledNativeCountedMessage::LocalHotMissKeys || nominal >= PooledNativeCountedMessage::Count)
        throw std::invalid_argument("MissKeys requires its original request-lane completion frame");
    requireCompletionSpans(device, extent, interval, error, payload, failureCode);
}

void PooledNativeRequestCompletionFrame::require(PooledNativeReaderMessageProfile profile, DeviceId device,
    const PooledNativeTransferSpan& extent, const PooledNativeTransferSpan& payload) const
{
    if (profile != PooledNativeReaderMessageProfile::CompactNative)
        throw std::invalid_argument("Request-lane completion requires the original CompactNative profile");
    requireCompletionSpans(device, extent, request_identity, error, payload, failureCode);
}
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
PooledNativeTransferSpan PooledNativeTransferSpan::subspan(std::size_t relativeOffset, std::size_t bytes) const
{
    if (!bytes || relativeOffset > bytes_ || bytes > bytes_ - relativeOffset ||
        relativeOffset > std::numeric_limits<std::size_t>::max() - offset_)
        throw std::out_of_range("Native pooled field exceeds its original declared span");
    // Reuse the existing root-bound check without exposing or replacing its
    // physical owner. Unlike resized(), the requested field cannot grow to
    // otherwise unused root capacity outside this declaration.
    auto result = resized(bytes_);
    result.offset_ += relativeOffset;
    result.bytes_ = bytes;
    return result;
}

bool PooledNativeTransferSpan::sameOwner(const PooledNativeTransferSpan& other) const noexcept
{
    if (owner_.index() != other.owner_.index()) return false;
    return std::visit([&](const auto& owner) {
        const auto& right = std::get<std::decay_t<decltype(owner)>>(other.owner_);
        return owner.get() == right.get() && !owner.owner_before(right) && !right.owner_before(owner);
    }, owner_);
}
void PooledNativeTransferSpan::requireDisjoint(const PooledNativeTransferSpan& other) const
{
    if (!bytes_ || !other.bytes_ || offset_ > std::numeric_limits<std::size_t>::max() - bytes_ ||
        other.offset_ > std::numeric_limits<std::size_t>::max() - other.bytes_)
        throw std::invalid_argument("Native pooled disjoint fields require bounded positive subranges");
    if (owner_.index() != other.owner_.index()) return;
    std::visit([&](const auto& owner) {
        const auto& right = std::get<std::decay_t<decltype(owner)>>(other.owner_);
        if (!owner || !right)
            throw std::invalid_argument("Native pooled disjoint fields lost an original physical root");
        if (owner.get() != right.get()) return;
        if (owner.owner_before(right) || right.owner_before(owner))
            throw std::invalid_argument("Native pooled fields alias another physical ownership control block");
        if (offset_ < other.offset_ + other.bytes_ && other.offset_ < offset_ + bytes_)
            throw std::invalid_argument("Native pooled completion metadata overlaps its original payload or another field");
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
    NativeLocalTPContextBorrow cohort, PreparedGPUContextUsePermit original,
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
    NativeLocalTPContextBorrow cohort, PreparedGPUContextUsePermit original,
    CapturedTransferEndpoint side, int producerCoordinate, int consumerCoordinate, CapturedTransferMessage message,
    PooledNativeTransferSpan payload)
{
    const auto participant=side==CapturedTransferEndpoint::Producer?producerCoordinate:consumerCoordinate;
    NativeCompletionAuthority authority;
    {
        const auto use=cohort.acquire();auto& context=use.context();
        if(isGraphCaptureActive() || !NativeByteEdge(producerCoordinate,consumerCoordinate).contains(context.degree(),participant) ||
            !message.key || !message.bytes || (side!=CapturedTransferEndpoint::Producer && side!=CapturedTransferEndpoint::Consumer))
            throw std::invalid_argument("Native pooled transfer requires an original cold addressed fixed declaration");
        if(context.devices().at(participant).toLocalDeviceId()!=original.device() ||
            context.backend()!=(original.device().is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL))
            throw std::invalid_argument("Native pooled endpoint changed its original communicator coordinate/backend");
        authority=context.nativeCompletionAuthority(participant);
    }
    payload.require(original.device());
    if(message.bytes>payload.bytes())throw std::out_of_range("Native pooled message exceeds its original physical subrange");
    PooledImmutableNativeTransfer result(std::move(cohort),std::move(original),side,
        producerCoordinate,consumerCoordinate,participant,message,std::move(payload));
    result.authority_=authority;
    if(!result.authority_.valid())throw std::logic_error("Native pooled fixed endpoint lacks its original native communicator authority");
    result.requireCurrent();
    result.sealStorage();
    return result;
}

PooledImmutableNativeTransfer PooledImmutableNativeTransfer::withPrefix(
    PooledNativeCountedMessage nominal, PooledNativeCompletionLayout layout,
    std::shared_ptr<MappedHostTransferRegion> operandBacking, std::shared_ptr<DeviceTransferBuffer> receipts,
    PooledNativeTransferSpan extent, PooledNativeCompletionFrame frame) const
{
    if (nominal == PooledNativeCountedMessage::LocalHotMissKeys)
        throw std::invalid_argument("MissKeys cannot use a pool-interval completion frame");
    return withCountedFrame(nominal, layout, std::move(operandBacking), std::move(receipts),
        std::move(extent), std::move(frame));
}

PooledImmutableNativeTransfer PooledImmutableNativeTransfer::withRequestPrefix(
    PooledNativeCompletionLayout layout, std::shared_ptr<MappedHostTransferRegion> operandBacking,
    std::shared_ptr<DeviceTransferBuffer> receipts, PooledNativeTransferSpan extent,
    PooledNativeRequestCompletionFrame frame) const
{
    return withCountedFrame(PooledNativeCountedMessage::LocalHotMissKeys, layout,
        std::move(operandBacking), std::move(receipts), std::move(extent), std::move(frame));
}

PooledImmutableNativeTransfer PooledImmutableNativeTransfer::withCountedFrame(
    PooledNativeCountedMessage nominal, PooledNativeCompletionLayout layout,
    std::shared_ptr<MappedHostTransferRegion> operandBacking, std::shared_ptr<DeviceTransferBuffer> receipts,
    PooledNativeTransferSpan extent, CountedFrame frame) const
{
    requireCurrent();
    if (isGraphCaptureActive() || operand_published_ || extent_ || !operandBacking || !receipts ||
        layout.degree()!=static_cast<std::size_t>(cohort_.acquire().context().degree()) ||
        !layout.matchesStorageBytes(operandBacking->sizeBytes(), receipts->sizeBytes(),participant_) ||
        receipts->device() != device())
        throw std::invalid_argument("Counted native pooled binding requires original cold admitted completion roots");
    const bool request = std::holds_alternative<PooledNativeRequestCompletionFrame>(frame);
    if (request != (nominal == PooledNativeCountedMessage::LocalHotMissKeys))
        throw std::invalid_argument("Native pooled counted message changed its original completion purpose");
    std::visit([&](const auto& original) {
        using Frame = std::decay_t<decltype(original)>;
        if constexpr (std::is_same_v<Frame, PooledNativeRequestCompletionFrame>)
            original.require(layout.readerProfile(), device(), extent, payload_);
        else original.require(nominal, device(), extent, payload_);
    }, frame);
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
    result.layout_ = PooledNativeCompletionBindingLayout{layout.degree(),layout.coordinator(),
        layout.operandOffset(nominal,participant_,NativeByteEdge(producer_,consumer_).peer(participant_)),
        layout.receiptOffset(nominal,participant_,NativeByteEdge(producer_,consumer_).peer(participant_)),layout.readerProfile()};
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
        (frame_ && std::holds_alternative<PooledNativeRequestCompletionFrame>(*frame_)) ||
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
        (frame_ && std::holds_alternative<PooledNativeRequestCompletionFrame>(*frame_)) ||
        side_ != CapturedTransferEndpoint::Consumer)
        throw std::invalid_argument("Native pooled receive slice requires its original cold acquired extent");
    auto result = *this;
    result.slice_message_ = true;
    return result;
}

const PooledNativeTransferSpan& PooledImmutableNativeTransfer::frameIdentity() const
{
    if (!frame_) throw std::logic_error("Counted native pooled transfer has no original completion purpose");
    return std::visit([](const auto& frame) -> const PooledNativeTransferSpan& {
        using Frame = std::decay_t<decltype(frame)>;
        if constexpr (std::is_same_v<Frame, PooledNativeRequestCompletionFrame>) return frame.request_identity;
        else return frame.interval;
    }, *frame_);
}

const PooledNativeTransferSpan& PooledImmutableNativeTransfer::frameError() const
{
    if (!frame_) throw std::logic_error("Counted native pooled transfer lost its original error field");
    return std::visit([](const auto& frame) -> const PooledNativeTransferSpan& { return frame.error; }, *frame_);
}

std::int32_t PooledImmutableNativeTransfer::frameFailureCode() const
{
    if (!frame_) throw std::logic_error("Counted native pooled transfer lost its original failure code");
    return std::visit([](const auto& frame) { return frame.failureCode; }, *frame_);
}

void* PooledImmutableNativeTransfer::pointer(const PooledNativeTransferSpan& span, std::size_t alignment) const
{
    return span.pointer(device(), alignment);
}

void PooledImmutableNativeTransfer::requireCurrent() const
{
    const auto use=cohort_.acquire();
    const auto& context=use.context();
    if (!NativeByteEdge(producer_,consumer_).contains(context.degree(),participant_))
        throw std::logic_error("Native pooled transfer lost its original addressed owner");
    if(context.devices().at(participant_).toLocalDeviceId()!=device() ||
        context.backend()!=(device().is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL))
        throw std::logic_error("Native pooled endpoint lost its original communicator coordinate/backend");
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
            extent_ ? &*extent_ : nullptr, frame_ ? &frameIdentity() : nullptr,
            frame_ ? &frameError() : nullptr, slice_ ? &*slice_ : nullptr};
        for (std::size_t i = 0; i < spans.size(); ++i) if (spans[i]) {
            if (spans[i]->pointer(device(), spanAlignments[i]) != addresses_[i])
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
        extent_ ? &*extent_ : nullptr, frame_ ? &frameIdentity() : nullptr,
        frame_ ? &frameError() : nullptr, slice_ ? &*slice_ : nullptr};
    std::array<TensorBase*, 5> roots{};
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>, 5> borrowed{};
    std::array<void*, 5> addresses{};
    std::size_t count = 0;
    for (std::size_t i = 0; i < spans.size(); ++i) if (spans[i]) {
        addresses[i] = spans[i]->pointer(device(), spanAlignments[i]);
        if (const auto tensor = spans[i]->tensor()) {
            roots[count++] = tensor.get();
            for (std::size_t prior = 0; prior < i; ++prior)
                if (spans[prior] && spans[i]->sameOwner(*spans[prior])) { borrowed[i] = borrowed[prior]; break; }
            if (!borrowed[i]) borrowed[i] = TransferEngine::instance().retainCapturedDeviceStorage(
                *tensor, device(), tensor->size_bytes());
        }
    }
    if (frame_)
    {
        // Distinct owner objects must not conceal overlapping native addresses.
        // This compares immutable prepared addresses only, never device values.
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t prior = 0; prior < i; ++prior)
            {
                const auto left = reinterpret_cast<std::uintptr_t>(addresses[i]);
                const auto right = reinterpret_cast<std::uintptr_t>(addresses[prior]);
                if (spans[i]->bytes() > std::numeric_limits<std::uintptr_t>::max() - left ||
                    spans[prior]->bytes() > std::numeric_limits<std::uintptr_t>::max() - right ||
                    (left < right + spans[prior]->bytes() && right < left + spans[i]->bytes()))
                    throw std::invalid_argument("Native pooled completion lost its disjoint original prepared fields");
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
        auto* identity = static_cast<std::uint64_t*>(pointer(frameIdentity(), alignof(std::uint64_t)));
        auto* error = static_cast<std::int32_t*>(pointer(frameError(), alignof(std::int32_t)));
        auto* receipt = static_cast<ncclLlaminarByteCompletion*>(receipts_->mutableDeviceData(
            layout_->receipt_offset));
        // The same native three-pointer ABI borrows the purpose-tagged original
        // words directly; request identity never becomes a synthetic pool epoch.
        const ncclLlaminarCompletionOperand operand{identity, identity + 1, identity + 2, error,
            receipt, nullptr, message_.key, frameFailureCode(), 0};
        auto* target = static_cast<std::byte*>(operands_->mutableHostData()) +
            layout_->operand_offset;
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
    const auto use=cohort_.acquire();
    auto& context = use.context();
    auto bytes = NativeCollectiveBytes::fixed(message_.bytes);
    if (extent_) {
        const auto* operand = reinterpret_cast<const ncclLlaminarCompletionOperand*>(
            static_cast<std::byte*>(operands_->deviceAlias(device())) +
            layout_->operand_offset);
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
        static_cast<const ncclLlaminarByteCompletion*>(receipts_->deviceData(layout_->receipt_offset)));
}

std::uint64_t PooledImmutableNativeTransfer::captureIdentity() const
{
    requireCurrent();
    std::uint64_t value = 1469598103934665603ULL;
    const auto mix = [&](std::uint64_t word) { value = (value ^ word) * 1099511628211ULL; };
    const auto use=cohort_.acquire();
    mix(reinterpret_cast<std::uintptr_t>(&use.context()));
    mix(original_.generation()); mix(static_cast<std::uint64_t>(device().type)); mix(device().ordinal);
    mix(producer_); mix(consumer_); mix(participant_); mix(message_.key); mix(message_.bytes);
    mix(reinterpret_cast<std::uintptr_t>(authority_.binding().abortFlag));
    mix(reinterpret_cast<std::uintptr_t>(authority_.binding().deviceCommunicator));
    const auto span = [&](const PooledNativeTransferSpan& item) { mix(item.identity()); };
    span(payload_);
    if (extent_) {
        span(*extent_); span(frameIdentity()); span(frameError());
        mix(frame_->index()); mix(static_cast<std::uint64_t>(*nominal_));
        mix(static_cast<std::uint64_t>(layout_->profile));
        mix(layout_->degree); mix(layout_->coordinator);
        mix(static_cast<std::uint64_t>(frameFailureCode()));
        mix(reinterpret_cast<std::uintptr_t>(operands_.get()));
        mix(reinterpret_cast<std::uintptr_t>(receipts_.get()));
        mix(layout_->operand_offset);
        mix(layout_->receipt_offset);
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
