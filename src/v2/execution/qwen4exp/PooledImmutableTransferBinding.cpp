/**
 * @file PooledImmutableTransferBinding.cpp
 * @brief Cold explicit transport binding and exact PMA metadata for ordered pooled PLE operations.
 *
 * Native execution uses the actual Rank-issued TP communicator and original
 * prepared workers. The existing TransferEngine remains the cross-vendor path.
 * No enqueue chooses a route, allocates, reads a device count or retries.
 */
#include "PooledImmutableTransferBinding.h"
#include "memory/CPUStorageRoot.h"
#include "backends/GPUDeviceContextPool.h"
#include <new>
#include <cstring>

namespace llaminar2::qwen4exp {
namespace {
/** @return True only for original source-relative descriptor roles in the existing protocol. */
bool sourceSliceMessage(PooledNativeCountedMessage message)
{
    return message == PooledNativeCountedMessage::PinCommands || message == PooledNativeCountedMessage::FreeDecisions ||
        message == PooledNativeCountedMessage::VictimDecisions || message == PooledNativeCountedMessage::ReadRequests;
}
/** @return Exact existing CPU allocator request for the retained metadata owner object. */
std::size_t objectBytes()
{
    return CPUStorageRoot<std::byte>(sizeof(PooledImmutableNativeCompletionStorage),
        PhysicalMemoryOwner::ActivationTransportStaging, DeviceId::cpu()).physicalBytes();
}
}
PhysicalMemoryPlan PooledImmutableNativeCompletionStorage::memoryBOM(PhysicalMemoryResource host,
    std::span<const PhysicalMemoryResource> participants, PooledNativeCompletionLayout layout)
{
    if(!host.valid() || host.device!=DeviceId::cpu() || participants.size()!=layout.degree())
        throw std::invalid_argument("Native pooled metadata requires its exact original participant resources");
    for(std::size_t i=0;i<participants.size();++i){
        const auto& value=participants[i];
        if(!value.valid() || !value.device.is_gpu() || value.world_rank!=host.world_rank ||
            value.device.type!=participants.front().device.type)
            throw std::invalid_argument("Native pooled metadata changed original homogeneous rank resources");
        for(std::size_t prior=0;prior<i;++prior)if(participants[prior].device==value.device)
            throw std::invalid_argument("Native pooled metadata repeats an original physical participant");
    }
    PhysicalMemoryPlanBuilder plan;
    plan.add(host, PhysicalMemoryOwner::ActivationTransportStaging, objectBytes());
    plan.add(host, PhysicalMemoryOwner::ActivationTransportStaging,
        layout.operandAllocationBytes());
    for (std::size_t i=0;i<participants.size();++i)
        plan.add(participants[i], PhysicalMemoryOwner::ActivationTransportStaging, layout.receiptBytesPerParticipant(i));
    return plan.build();
}
std::shared_ptr<const PooledImmutableNativeCompletionStorage> PooledImmutableNativeCompletionStorage::create(
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort,
    Participants participants,
    Frames frames, PooledNativeCompletionLayout layout)
{
    if (isGraphCaptureActive() || !cohort || std::size_t(cohort->degree())!=layout.degree())
        throw std::invalid_argument("Native pooled metadata requires its actual cold original addressed cohort");
    if(!participants[0])throw std::invalid_argument("Native pooled metadata omitted its original participant");
    const auto memory=participants[0]->memory();
    if(!memory)throw std::invalid_argument("Native pooled metadata lost its original PMA");
    std::array<DeviceId,kMaximumPooledNativeParticipants> devices;
    std::array<NativeCompletionAuthority,kMaximumPooledNativeParticipants> authorities;
    for(std::size_t i=0;i<layout.degree();++i){
        if(!participants[i] || participants[i]->memory().get()!=memory.get() ||
            participants[i]->memory().owner_before(memory) || memory.owner_before(participants[i]->memory()))
            throw std::invalid_argument("Native pooled metadata changed its original PMA/control block");
        devices[i]=participants[i]->device();authorities[i]=cohort->context().nativeCompletionAuthority(int(i));
        cohort->requireConsumer(devices[i],int(i));
        const auto current=GPUDeviceContextPool::instance().retainExistingPreparedContext(devices[i],memory);
        participants[i]->requireWorker(current.worker());
        if(current.generation()!=participants[i]->generation() || participants[i]->worker()->ownsCurrentThread() || !authorities[i].valid())
            throw std::logic_error("Native pooled metadata lost its original off-worker generation/communicator");
    }
    for(std::size_t i=layout.degree();i<participants.size();++i)if(participants[i] || frames[i])
        throw std::invalid_argument("Native pooled metadata contains a participant outside its original cohort");
    auto object = CPUStorageRoot<std::byte>(sizeof(PooledImmutableNativeCompletionStorage),
        PhysicalMemoryOwner::ActivationTransportStaging, DeviceId::cpu(), CPUStoragePlacement::local(),
        CPUStorageInitialization::Uninitialized).allocate(memory);
    auto mappedClaim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
        layout.operandAllocationBytes());
    std::array<PhysicalMemoryAllocationLease,kMaximumPooledNativeParticipants> claims;
    auto operands = TransferEngine::instance().allocateMappedHostRegion(layout.operandBytes(), std::span(devices.data(),layout.degree()));
    // Only cold immutable metadata is cleared. Later captures must match these
    // original bindings exactly instead of rebinding an earlier graph's slot.
    std::memset(operands->mutableHostData(), 0, layout.operandBytes());
    std::array<std::shared_ptr<DeviceTransferBuffer>,kMaximumPooledNativeParticipants> receipts;
    for (std::size_t i = 0; i < layout.degree(); ++i) {
        claims[i] = memory->claimNewAllocation(devices[i], PhysicalMemoryOwner::ActivationTransportStaging,
            layout.receiptBytesPerParticipant(i));
        participants[i]->worker()->submitAndWait([&] {
            receipts[i] = TransferEngine::instance().allocateDeviceTransferBuffer(layout.receiptBytesPerParticipant(i), devices[i]);
        });
    }
    static_assert(alignof(PooledImmutableNativeCompletionStorage) <= AlignedVector<std::byte>::ALIGNMENT);
    auto* address = new(object.data()) PooledImmutableNativeCompletionStorage(std::move(cohort), std::move(participants),
        std::move(frames), layout, std::move(authorities), std::move(mappedClaim), std::move(claims), std::move(operands), std::move(receipts));
    return std::shared_ptr<const PooledImmutableNativeCompletionStorage>(address,
        [storage = std::move(object)](const PooledImmutableNativeCompletionStorage* value) mutable {
            value->~PooledImmutableNativeCompletionStorage(); storage = {};
        });
}
int PooledImmutableNativeCompletionStorage::coordinate(DeviceId device) const
{
    requireCurrent();
    for (std::size_t i = 0; i < layout_.degree(); ++i) if (participants_[i]->device() == device) return i;
    throw std::invalid_argument("Native pooled binding is absent from its original communicator membership");
}
const PreparedGPUContextUsePermit& PooledImmutableNativeCompletionStorage::original(DeviceId device) const
{
    return *participants_[coordinate(device)];
}
void PooledImmutableNativeCompletionStorage::requireCurrent() const
{
    if (retired_.load(std::memory_order_acquire))
        throw std::logic_error("Native pooled runtime bindings have retired with their original parents");
    if (!cohort_ || std::size_t(cohort_->degree())!=layout_.degree() || !operands_)
        throw std::logic_error("Native pooled metadata lost its original cohort/backing");
    const auto memory = participants_[0]->memory();
    for (std::size_t i = 0; i < layout_.degree(); ++i) {
        cohort_->requireConsumer(participants_[i]->device(), i);
        const auto current = GPUDeviceContextPool::instance().retainExistingPreparedContext(participants_[i]->device(), memory);
        participants_[i]->requireWorker(current.worker());
        if (current.generation() != participants_[i]->generation() ||
            participants_[i]->memory().get() != memory.get() || participants_[i]->memory().owner_before(memory) ||
            memory.owner_before(participants_[i]->memory()) || !receipts_[i] ||
            receipts_[i]->device() != participants_[i]->device() ||
            !layout_.matchesStorageBytes(operands_->sizeBytes(), receipts_[i]->sizeBytes(),i) ||
            mapped_claim_.bytes() != layout_.operandAllocationBytes() ||
            receipt_claims_[i].bytes() != layout_.receiptBytesPerParticipant(i) ||
            !operands_->hasDevice(participants_[i]->device()) ||
            !authorities_[i].valid())
            throw std::logic_error("Native pooled metadata changed its original generation/PMA/native resources");
    }
}
int PooledImmutableNativeCompletionStorage::requireOriginalBinding(
    const PooledImmutableNativeTransfer& binding) const
{
    requireCurrent();
    if (!binding.cohort_ || binding.cohort_.get() != cohort_.get() ||
        binding.cohort_.owner_before(cohort_) || cohort_.owner_before(binding.cohort_))
        throw std::invalid_argument("Native pooled counted binding belongs to another original cohort");
    const auto participant = coordinate(binding.device());
    const auto& original = *participants_[participant];
    const auto memory = original.memory();
    if (binding.original_.memory().get() != memory.get() ||
        binding.original_.memory().owner_before(memory) || memory.owner_before(binding.original_.memory()))
        throw std::invalid_argument("Native pooled counted binding changed its original physical authority");
    original.requireWorker(binding.original_.worker());
    if (binding.original_.generation() != original.generation())
        throw std::logic_error("Native pooled counted binding changed its original worker generation");
    return participant;
}

PooledImmutableNativeTransfer PooledImmutableNativeCompletionStorage::counted(PooledImmutableNativeTransfer binding,
    PooledNativeCountedMessage message, PooledNativeTransferSpan extent,
    std::optional<PooledNativeCompletionFrame> frame) const
{
    const auto participant = requireOriginalBinding(binding);
    if (!frame) frame = frames_[participant];
    if (!frame) throw std::invalid_argument("Native pooled completion requires its actual original local frame/error authority");
    auto result = binding.withPrefix(message, layout_, operands_, receipts_[participant], std::move(extent), std::move(*frame));
    if (binding.role() == CapturedTransferEndpoint::Consumer && sourceSliceMessage(message))
        result = result.withReceivedSourceSlice();
    return result;
}

PooledImmutableNativeTransfer PooledImmutableNativeCompletionStorage::countedRequest(
    PooledImmutableNativeTransfer binding, PooledNativeTransferSpan extent,
    PooledNativeRequestCompletionFrame frame) const
{
    const auto participant = requireOriginalBinding(binding);
    // Request completion is always explicit. The optional source/reader pool
    // frames remain response-only and cannot supply this request identity.
    return binding.withRequestPrefix(layout_, operands_, receipts_[participant],
        std::move(extent), std::move(frame));
}

std::uint64_t PooledImmutableTransferBinding::captureIdentity() const
{
    if (completion_) completion_->requireCurrent();
    const auto identity = std::visit([](const auto& binding) { return binding.captureIdentity(); }, binding_);
    return nativeCollective() ? ((identity ^ 0x4e434c5f504c4555ULL) * 1099511628211ULL) | 1u : identity;
}
CapturedTransferMessage PooledImmutableTransferBinding::message() const noexcept
{ return std::visit([](const auto& binding) { return binding.message(); }, binding_); }
CapturedTransferExtentSource PooledImmutableTransferBinding::extentSource() const noexcept
{ return std::visit([](const auto& binding) { return binding.extentSource(); }, binding_); }
bool PooledImmutableTransferBinding::hasSourceSlice() const noexcept
{ return std::visit([](const auto& binding) { return binding.hasSourceSlice(); }, binding_); }
DeviceId PooledImmutableTransferBinding::device() const noexcept
{ return std::visit([](const auto& binding) { return binding.device(); }, binding_); }
CapturedTransferEndpoint PooledImmutableTransferBinding::role() const noexcept
{ return std::visit([](const auto& binding) { return binding.role(); }, binding_); }
bool PooledImmutableTransferBinding::nativeCollective() const noexcept
{ return std::holds_alternative<PooledImmutableNativeTransfer>(binding_); }
PooledImmutableTransferBinding PooledImmutableTransferBinding::withDeviceCounted(std::shared_ptr<TensorBase> root,
    std::size_t offset, PooledNativeCountedMessage message) const
{
    if (const auto* original = std::get_if<CapturedTransferBinding>(&binding_))
        return TransferEngine::instance().bindDeviceCountedTransfer(*original, std::move(root), offset);
    if (!completion_) throw std::logic_error("Native pooled counted binding lost its original metadata owner");
    if (role() == CapturedTransferEndpoint::Producer && sourceSliceMessage(message))
        throw std::invalid_argument("Native pooled source-slice producer cannot substitute a prefix descriptor");
    auto original = completion_->counted(std::get<PooledImmutableNativeTransfer>(binding_), message,
        PooledNativeTransferSpan(std::move(root), offset, sizeof(std::uint64_t)));
    return {std::move(original), completion_};
}
PooledImmutableTransferBinding PooledImmutableTransferBinding::withDeviceSliced(std::shared_ptr<TensorBase> root,
    std::size_t offset, std::size_t sourceBytes, std::uint64_t sourceOrigin, PooledNativeCountedMessage message) const
{
    if (const auto* original = std::get_if<CapturedTransferBinding>(&binding_))
        return TransferEngine::instance().bindDeviceSlicedTransfer(*original, std::move(root), offset, sourceBytes, sourceOrigin);
    if (!completion_ || role() != CapturedTransferEndpoint::Producer || !sourceSliceMessage(message))
        throw std::invalid_argument("Native pooled slice requires its original producer/nominal metadata owner");
    const PooledNativeTransferSpan descriptor(root, offset, sizeof(NativeByteSourceSlice));
    auto original = completion_->counted(std::get<PooledImmutableNativeTransfer>(binding_), message,
        PooledNativeTransferSpan(std::move(root), offset + offsetof(NativeByteSourceSlice, live_bytes), sizeof(std::uint64_t)));
    original = original.withSourceSlice(descriptor, sourceBytes, sourceOrigin);
    return {std::move(original), completion_};
}
void PooledImmutableTransferBinding::prepare(void* stream) const
{
    if (completion_) completion_->requireCurrent();
    if (auto* original = std::get_if<PooledImmutableNativeTransfer>(&binding_)) original->prepare(stream);
}
void PooledImmutableTransferBinding::enqueue(void* stream) const
{
    if (completion_) completion_->requireCurrent();
    if (const auto* original = std::get_if<CapturedTransferBinding>(&binding_))
        TransferEngine::instance().enqueueCapturedTransfer(*original, stream);
    else std::get<PooledImmutableNativeTransfer>(binding_).enqueue(stream);
}
CapturedTransferReceiveCompletion PooledImmutableTransferBinding::mappedCompletion(void* stream) const
{
    const auto* original = std::get_if<CapturedTransferBinding>(&binding_);
    if (!original) throw std::logic_error("Native pooled transfer cannot manufacture mapped completion");
    return TransferEngine::instance().capturedReceiveCompletion(*original, stream);
}
NativeReceiveCompletionView PooledImmutableTransferBinding::nativeCompletion() const
{
    if (completion_) completion_->requireCurrent();
    const auto* original = std::get_if<PooledImmutableNativeTransfer>(&binding_);
    if (!original) throw std::logic_error("Mapped pooled transfer cannot manufacture native completion");
    return original->completion();
}
PooledImmutableTransferBinding PooledImmutableTransferBinding::withNativeCounted(
    PooledNativeTransferSpan extent, PooledNativeCountedMessage message,
    PooledNativeCompletionFrame frame) const
{
    const auto* original = std::get_if<PooledImmutableNativeTransfer>(&binding_);
    if (!original || !completion_)
        throw std::logic_error("Native pooled count cannot replace an original cross-vendor binding");
    return {completion_->counted(*original, message, std::move(extent), std::move(frame)), completion_};
}
PooledImmutableTransferBinding PooledImmutableTransferBinding::withNativeRequestCounted(
    PooledNativeTransferSpan extent, PooledNativeRequestCompletionFrame frame) const
{
    const auto* original = std::get_if<PooledImmutableNativeTransfer>(&binding_);
    if (!original || !completion_)
        throw std::logic_error("Native pooled request count cannot replace an original cross-vendor binding");
    return {completion_->countedRequest(*original, std::move(extent), std::move(frame)), completion_};
}

const CapturedTransferBinding& PooledImmutableTransferBinding::captured() const
{
    const auto* original = std::get_if<CapturedTransferBinding>(&binding_);
    if (!original) throw std::logic_error("Native pooled transfer has no captured-channel binding");
    return *original;
}
void PooledImmutableTransferBinding::bindOriginalInvocation(PooledImmutableOwnerPreparedStorage& owner) const
{
    if (const auto* original = std::get_if<CapturedTransferBinding>(&binding_)) owner.bindInvocationChannel(*original);
    else {
        if (!completion_ || role() != CapturedTransferEndpoint::Consumer || device() != owner.plan().parameters().device)
            throw std::logic_error("Native pooled invocation changed its original consumer/physical owner");
        owner.bindNativeInvocation(completion_->cohort(), completion_->coordinate(device()));
    }
}
void PooledImmutableTransferBinding::releaseOriginalInvocation(PooledImmutableOwnerPreparedStorage& owner) const
{
    if (const auto* original = std::get_if<CapturedTransferBinding>(&binding_)) owner.releaseInvocationChannel(*original);
    else {
        if (!completion_) throw std::logic_error("Native invocation retirement lost its original metadata owner");
        owner.releaseNativeInvocation(completion_->cohort(), completion_->coordinate(device()));
    }
}
PooledImmutableTransferChannel PooledImmutableTransferChannel::native(
    std::shared_ptr<const PooledImmutableNativeCompletionStorage> original, DeviceId producer, DeviceId consumer, std::size_t maximumBytes)
{
    if (isGraphCaptureActive() || !original || !maximumBytes)
        throw std::invalid_argument("Native pooled channel requires its original cold metadata owner and bounded extent");
    original->requireCurrent();
    PooledImmutableTransferChannel result;
    result.producer_ = original->coordinate(producer); result.consumer_=original->coordinate(consumer);
    (void)NativeByteEdge(result.producer_,result.consumer_); result.maximum_ = maximumBytes; result.native_ = std::move(original);
    return result;
}
DeviceId PooledImmutableTransferChannel::endpointDevice(CapturedTransferEndpoint side) const
{
    if (side != CapturedTransferEndpoint::Producer && side != CapturedTransferEndpoint::Consumer)
        throw std::invalid_argument("Pooled transfer has an invalid original endpoint");
    if (captured_) return captured_->endpointDevice(side);
    if (!native_) throw std::logic_error("Pooled transfer has no original channel owner");
    const int coordinate = side == CapturedTransferEndpoint::Producer ? producer_ : consumer_;
    return native_->cohort()->context().devices()[coordinate].toLocalDeviceId();
}
PooledImmutableTransferBinding PooledImmutableTransferChannel::bind(CapturedTransferEndpoint side,
    CapturedTransferMessage message, std::shared_ptr<TensorBase> root, std::size_t offset) const
{
    if (captured_) return TransferEngine::instance().bindCapturedTransfer(captured_, side, message, std::move(root), offset);
    if (!native_ || !message.fits(maximum_))
        throw std::invalid_argument("Native pooled endpoint exceeds its original directed message envelope");
    const auto device = endpointDevice(side);
    auto native = PooledImmutableNativeTransfer::fixed(native_->cohort(), native_->original(device), side, producer_, consumer_, message,
        PooledNativeTransferSpan(std::move(root), offset, message.bytes));
    return {std::move(native), native_};
}
PooledImmutableTransferBinding PooledImmutableTransferChannel::bind(CapturedTransferEndpoint side,
    CapturedTransferMessage message, std::shared_ptr<DeviceTransferBuffer> root, std::size_t offset) const
{
    if (captured_)
        return TransferEngine::instance().bindCapturedTransfer(captured_, side, message, std::move(root), offset);
    if (!native_ || !message.fits(maximum_))
        throw std::invalid_argument("Native pooled protocol endpoint exceeds its original message envelope");
    const auto device = endpointDevice(side);
    auto binding = PooledImmutableNativeTransfer::fixed(native_->cohort(), native_->original(device), side,
        producer_, consumer_, message, PooledNativeTransferSpan(std::move(root), offset, message.bytes));
    return {std::move(binding), native_};
}
PooledImmutableTransferBinding PooledImmutableTransferChannel::bind(CapturedTransferEndpoint side,
    CapturedTransferMessage message,PooledNativeTransferSpan span)const
{
    if(!message.fits(span.bytes()))
        throw std::invalid_argument("Pooled endpoint exceeds its exact original declared span");
    if(captured_){
        return std::visit([&](const auto& root){return bind(side,message,root,span.offset());},span.owner_);
    }
    if(!native_ || !message.fits(maximum_))throw std::invalid_argument("Pooled span endpoint has no original bounded transport");
    const auto device=endpointDevice(side);
    auto binding=PooledImmutableNativeTransfer::fixed(native_->cohort(),native_->original(device),side,
        producer_,consumer_,message,span.subspan(0,message.bytes));
    return {std::move(binding),native_};
}

} // namespace llaminar2::qwen4exp
