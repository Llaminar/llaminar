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
}
std::optional<NativeLocalTPContextBorrow> PooledImmutableTransferBinding::originalNativeContext() const
{
    if(!nativeCollective())return std::nullopt;
    if(!completion_)throw std::logic_error("Native pooled binding lost its original completion authority");
    completion_->requireCurrent();
    return completion_->cohort_;
}
std::size_t PooledImmutableNativeCompletionStorage::participantOffset()
{
    constexpr auto alignment=alignof(Participant);
    return (sizeof(PooledImmutableNativeCompletionStorage)+alignment-1)/alignment*alignment;
}
std::size_t PooledImmutableNativeCompletionStorage::objectPayloadBytes(std::size_t degree)
{
    if(degree<2 || degree>std::size_t(std::numeric_limits<int>::max()) ||
        degree>(std::numeric_limits<std::size_t>::max()-participantOffset())/sizeof(Participant))
        throw std::invalid_argument("Native pooled metadata requires a representable original population");
    return participantOffset()+degree*sizeof(Participant);
}
PhysicalMemoryPlan PooledImmutableNativeCompletionStorage::memoryBOM(PhysicalMemoryResource host,
    std::span<const PhysicalMemoryResource> participants,PooledNativeCompletionLayout layout)
{
    if(!host.valid() || host.device!=DeviceId::cpu() || participants.size()!=layout.degree())
        throw std::invalid_argument("Native pooled metadata requires its exact original participant resources");
    for(std::size_t i=0;i<participants.size();++i) {
        const auto& value=participants[i];
        if(!value.valid() || !value.device.is_gpu() || value.world_rank!=host.world_rank ||
            value.device.type!=participants.front().device.type)
            throw std::invalid_argument("Native pooled metadata changed original homogeneous rank resources");
        for(std::size_t prior=0;prior<i;++prior)if(participants[prior].device==value.device)
            throw std::invalid_argument("Native pooled metadata repeats an original physical participant");
    }
    PhysicalMemoryPlanBuilder plan;
    plan.add(host,PhysicalMemoryOwner::ActivationTransportStaging,
        CPUStorageRoot<std::byte>(objectPayloadBytes(layout.degree()),
            PhysicalMemoryOwner::ActivationTransportStaging,DeviceId::cpu()).physicalBytes());
    plan.add(host,PhysicalMemoryOwner::ActivationTransportStaging,layout.operandAllocationBytes());
    for(std::size_t i=0;i<participants.size();++i)
        plan.add(participants[i],PhysicalMemoryOwner::ActivationTransportStaging,layout.receiptBytesPerParticipant(i));
    return plan.build();
}
std::shared_ptr<const PooledImmutableNativeCompletionStorage> PooledImmutableNativeCompletionStorage::create(
    NativeLocalTPContextBorrow cohort,std::unique_ptr<LocalTPContext> ownedContext,
    Participants participants,Frames frames,PooledNativeCompletionLayout layout)
{
    if(isGraphCaptureActive() || participants.size()!=layout.degree() || frames.size()!=layout.degree() || !participants[0])
        throw std::invalid_argument("Native pooled metadata requires its actual cold original population");
    const auto memory=participants[0]->memory();
    if(!memory)throw std::invalid_argument("Native pooled metadata lost its original PMA");
    {
        const auto use=cohort.acquire();const auto& context=use.context();
        if(std::size_t(context.degree())!=layout.degree() || context.devices().size()!=layout.degree() ||
            (ownedContext && !ownedContext->borrowOriginalNativeContext().sameOriginal(cohort)))
            throw std::invalid_argument("Native pooled metadata changed its original context ownership");
        for(std::size_t i=0;i<layout.degree();++i) {
            if(!participants[i] || participants[i]->memory().get()!=memory.get() ||
                participants[i]->memory().owner_before(memory) || memory.owner_before(participants[i]->memory()) ||
                context.devices()[i].toLocalDeviceId()!=participants[i]->device())
                throw std::invalid_argument("Native pooled metadata changed its original PMA/control block or member");
            const auto current=GPUDeviceContextPool::instance().retainExistingPreparedContext(participants[i]->device(),memory);
            participants[i]->requireWorker(current.worker());
            if(current.generation()!=participants[i]->generation() || participants[i]->worker()->ownsCurrentThread())
                throw std::logic_error("Native pooled metadata lost its original off-worker generation");
        }
    }
    auto backing=CPUStorageRoot<std::byte>(objectPayloadBytes(layout.degree()),
        PhysicalMemoryOwner::ActivationTransportStaging,DeviceId::cpu(),CPUStoragePlacement::local(),
        CPUStorageInitialization::Uninitialized).allocate(memory);
    static_assert(alignof(PooledImmutableNativeCompletionStorage)<=AlignedVector<std::byte>::ALIGNMENT);
    static_assert(alignof(Participant)<=AlignedVector<std::byte>::ALIGNMENT);
    auto* address=new(backing.data()) PooledImmutableNativeCompletionStorage(std::move(cohort),std::move(ownedContext),
        {reinterpret_cast<Participant*>(backing.data()+participantOffset()),layout.degree()},layout);
    auto result=std::shared_ptr<PooledImmutableNativeCompletionStorage>(address,
        [storage=std::move(backing)](PooledImmutableNativeCompletionStorage* value)mutable {
            value->~PooledImmutableNativeCompletionStorage();storage={};
        });
    std::vector<DeviceId> devices;devices.reserve(layout.degree());
    {
        const auto use=result->cohort_.acquire();auto& context=use.context();
        for(std::size_t i=0;i<layout.degree();++i) {
            auto* record=std::construct_at(&result->participants_[i]);++result->constructed_;
            record->original=std::move(participants[i]);record->frame=std::move(frames[i]);
            record->authority=context.nativeCompletionAuthority(int(i));
            record->positive_owner=layout.positiveOwner(i);record->reader=layout.reader(i);
            record->receipt_bytes=layout.receiptBytesPerParticipant(i);
            if(!record->authority.valid())throw std::logic_error("Native pooled member lost its completion authority");
            devices.push_back(record->original->device());
        }
    }
    if(layout.operandBytes()) {
        result->mapped_claim_=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationTransportStaging,
            layout.operandAllocationBytes());
        result->operands_=TransferEngine::instance().allocateMappedHostRegion(layout.operandBytes(),devices);
        // These are immutable cold operands, not live progress or capacity payloads.
        std::memset(result->operands_->mutableHostData(),0,layout.operandBytes());
    }
    for(auto& record:result->participants_)if(record.receipt_bytes) {
        record.receipt_claim=memory->claimNewAllocation(record.original->device(),
            PhysicalMemoryOwner::ActivationTransportStaging,record.receipt_bytes);
        record.original->worker()->submitAndWait([&] {
            record.receipt=TransferEngine::instance().allocateDeviceTransferBuffer(record.receipt_bytes,record.original->device());
        });
    }
    result->requireCurrent();return result;
}
PooledImmutableNativeCompletionStorage::~PooledImmutableNativeCompletionStorage()
{
    // Native graph borrowers release this shared owner before context destruction.
    // Each device root dies before its claim, then the single host root retires.
    while(constructed_)std::destroy_at(&participants_[--constructed_]);
}
PooledNativeCompletionLayout PooledImmutableNativeCompletionStorage::declaration()const
{
    std::vector<std::uint8_t> positive,readers;positive.reserve(participants_.size());readers.reserve(participants_.size());
    for(const auto& participant:participants_) {
        positive.push_back(participant.positive_owner);readers.push_back(participant.reader);
    }
    return {participants_.size(),coordinator_,positive,profile_,readers};
}
int PooledImmutableNativeCompletionStorage::coordinate(DeviceId device)const
{
    requireCurrent();
    for(std::size_t i=0;i<participants_.size();++i)if(participants_[i].original->device()==device)return int(i);
    throw std::invalid_argument("Native pooled binding is absent from its original communicator membership");
}
const PreparedGPUContextUsePermit& PooledImmutableNativeCompletionStorage::original(DeviceId device)const
{return *participants_[coordinate(device)].original;}
void PooledImmutableNativeCompletionStorage::requireCurrent()const
{
    if(retired_.load(std::memory_order_acquire))
        throw std::logic_error("Native pooled runtime bindings have retired with their original parents");
    const auto use=cohort_.acquire();const auto& context=use.context();
    if(constructed_!=participants_.size() || std::size_t(context.degree())!=participants_.size() ||
        context.devices().size()!=participants_.size() || bool(operands_)!=(operand_bytes_!=0) ||
        (operands_ && (operands_->sizeBytes()!=operand_allocation_bytes_ || mapped_claim_.bytes()!=operand_allocation_bytes_)))
        throw std::logic_error("Native pooled metadata lost its original cohort/backing");
    const auto memory=participants_[0].original->memory();
    for(std::size_t i=0;i<participants_.size();++i) {
        const auto& record=participants_[i];const auto& original=*record.original;
        const auto current=GPUDeviceContextPool::instance().retainExistingPreparedContext(original.device(),memory);
        original.requireWorker(current.worker());
        if(context.devices()[i].toLocalDeviceId()!=original.device() || current.generation()!=original.generation() ||
            original.memory().get()!=memory.get() || original.memory().owner_before(memory) || memory.owner_before(original.memory()) ||
            bool(record.receipt)!=(record.receipt_bytes!=0) ||
            (record.receipt && (record.receipt->device()!=original.device() || record.receipt->sizeBytes()!=record.receipt_bytes ||
                record.receipt_claim.bytes()!=record.receipt_bytes)) ||
            (operands_ && !operands_->hasDevice(original.device())) || !record.authority.valid())
            throw std::logic_error("Native pooled metadata changed its original generation/PMA/native resources");
    }
}
int PooledImmutableNativeCompletionStorage::requireOriginalBinding(const PooledImmutableNativeTransfer& binding)const
{
    requireCurrent();
    if(!binding.cohort_.sameOriginal(cohort_))
        throw std::invalid_argument("Native pooled counted binding belongs to another original cohort");
    const auto participant=coordinate(binding.device());const auto& original=*participants_[participant].original;
    const auto memory=original.memory();
    if(binding.original_.memory().get()!=memory.get() || binding.original_.memory().owner_before(memory) ||
        memory.owner_before(binding.original_.memory()))
        throw std::invalid_argument("Native pooled counted binding changed its original physical authority");
    original.requireWorker(binding.original_.worker());
    if(binding.original_.generation()!=original.generation())
        throw std::logic_error("Native pooled counted binding changed its original worker generation");
    return participant;
}
PooledImmutableNativeTransfer PooledImmutableNativeCompletionStorage::counted(PooledImmutableNativeTransfer binding,
    PooledNativeCountedMessage message,PooledNativeTransferSpan extent,std::optional<PooledNativeCompletionFrame> frame)const
{
    if(isGraphCaptureActive())throw std::logic_error("Native counted binding must be complete before capture");
    const auto participant=requireOriginalBinding(binding);
    if(!frame)frame=participants_[participant].frame;
    if(!frame)throw std::invalid_argument("Native completion requires its original local frame/error authority");
    auto result=binding.withPrefix(message,declaration(),operands_,participants_[participant].receipt,std::move(extent),std::move(*frame));
    if(binding.role()==CapturedTransferEndpoint::Consumer && sourceSliceMessage(message))result=result.withReceivedSourceSlice();
    return result;
}
PooledImmutableNativeTransfer PooledImmutableNativeCompletionStorage::countedRequest(PooledImmutableNativeTransfer binding,
    PooledNativeTransferSpan extent,PooledNativeRequestCompletionFrame frame)const
{
    if(isGraphCaptureActive())throw std::logic_error("Native request binding must be complete before capture");
    const auto participant=requireOriginalBinding(binding);
    return binding.withRequestPrefix(declaration(),operands_,participants_[participant].receipt,std::move(extent),std::move(frame));
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
        owner.bindNativeInvocation(completion_->cohort_, completion_->coordinate(device()));
    }
}
void PooledImmutableTransferBinding::releaseOriginalInvocation(PooledImmutableOwnerPreparedStorage& owner) const
{
    if (const auto* original = std::get_if<CapturedTransferBinding>(&binding_)) owner.releaseInvocationChannel(*original);
    else {
        if (!completion_) throw std::logic_error("Native invocation retirement lost its original metadata owner");
        owner.releaseNativeInvocation(completion_->cohort_, completion_->coordinate(device()));
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
    return native_->context().context().devices()[coordinate].toLocalDeviceId();
}
PooledImmutableTransferBinding PooledImmutableTransferChannel::bind(CapturedTransferEndpoint side,
    CapturedTransferMessage message, std::shared_ptr<TensorBase> root, std::size_t offset) const
{
    if (captured_) return TransferEngine::instance().bindCapturedTransfer(captured_, side, message, std::move(root), offset);
    if (!native_ || !message.fits(maximum_))
        throw std::invalid_argument("Native pooled endpoint exceeds its original directed message envelope");
    const auto device = endpointDevice(side);
    auto native = PooledImmutableNativeTransfer::fixed(native_->cohort_, native_->original(device), side, producer_, consumer_, message,
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
    auto binding = PooledImmutableNativeTransfer::fixed(native_->cohort_, native_->original(device), side,
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
    auto binding=PooledImmutableNativeTransfer::fixed(native_->cohort_,native_->original(device),side,
        producer_,consumer_,message,span.subspan(0,message.bytes));
    return {std::move(binding),native_};
}

} // namespace llaminar2::qwen4exp
