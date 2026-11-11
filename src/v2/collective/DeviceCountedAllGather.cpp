/**
 * @file DeviceCountedAllGather.cpp
 * @brief Compose original native counted broadcasts without host length observation.
 *
 * Setup freezes the rank-local topology and leases. Capture only binds arena
 * addresses and semantic message identity. Replay has no host count readback,
 * new allocation, stream synchronization, or alternate failure path.
 */
#include "DeviceCountedAllGather.h"
#include "LocalTPCollectiveInventory.h"
#include "ILocalTPContext.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/ITensor.h"
#include "tensors/TensorClasses.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
namespace
{
    /** @brief Checked physical geometry shared by admission and setup. */
    std::size_t multiply(std::size_t a, std::size_t b)
    {
        if (a && b > std::numeric_limits<std::size_t>::max() / a)
            throw std::overflow_error("Counted allgather geometry overflow");
        return a * b;
    }
}

DeviceCountedAllGatherTopology DeviceCountedAllGatherTopology::resolve(
    const RankInventory &inventory, std::span<const DeviceId> devices)
{
    if (inventory.rank < 0 || devices.size() < 2)
        throw std::invalid_argument("Counted topology requires an original rank and at least two GPUs");
    const auto route_count = multiply(devices.size(), devices.size());
    // Validate the complete matrix and membership before retaining any topology.
    const std::vector<DeviceId> members(devices.begin(), devices.end());
    (void)localTPPeerAccessCoverage(inventory, members);
    bool any = false, all = true;
    for (std::size_t i = 0; i < devices.size(); ++i)
        for (std::size_t j = i + 1; j < devices.size(); ++j)
        {
            const auto edge = localTPPeerAccessCoverage(inventory, {devices[i], devices[j]});
            if (edge == PeerAccessCoverage::Partial)
                throw std::invalid_argument("Counted topology rejects a one-way native peer edge");
            any |= edge == PeerAccessCoverage::Complete;
            all &= edge == PeerAccessCoverage::Complete;
        }
    DeviceCountedAllGatherTopology result;
    result.inventory_ = inventory;
    result.devices_.assign(devices.begin(), devices.end());
    result.routes_.resize(route_count, DeviceCountedAllGatherTransport::NativeCollective);
    result.coverage_ = all ? PeerAccessCoverage::Complete : any ? PeerAccessCoverage::Partial : PeerAccessCoverage::None;
    return result;
}

DeviceCountedAllGatherTransport DeviceCountedAllGatherTopology::payloadStorage(
    std::size_t producer, std::size_t consumer) const
{
    if (producer >= devices_.size() || consumer >= devices_.size() || producer == consumer)
        throw std::out_of_range("Counted topology edge is foreign or diagonal");
    return routes_[producer * devices_.size() + consumer];
}

DeviceCountedAllGatherMemory DeviceCountedAllGatherTopology::memoryFor(
    std::size_t participant, std::size_t capacity) const
{
    if (participant >= devices_.size()) throw std::out_of_range("Counted topology participant is foreign");
    if(!capacity || capacity>std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::invalid_argument("Native counted allgather requires positive representable packet capacity");
    return {0, NativeCountedAllGatherLayout(devices_.size()).bytes()};
}

bool DeviceCountedAllGatherTopology::sameRoutes(const DeviceCountedAllGatherTopology &other) const noexcept
{
    return this == &other || (worldRank() == other.worldRank() && devices_ == other.devices_ && routes_ == other.routes_ &&
        inventory_.p2p_cuda_count==other.inventory_.p2p_cuda_count && inventory_.p2p_cuda==other.inventory_.p2p_cuda &&
        inventory_.p2p_rocm_count==other.inventory_.p2p_rocm_count && inventory_.p2p_rocm==other.inventory_.p2p_rocm);
}

std::shared_ptr<DeviceCountedAllGather> DeviceCountedAllGather::create(
    PhysicalMemoryAuthority &authority, ILocalTPContext &original, DeviceId host_device,
    const DeviceCountedAllGatherTopology &topology,
    std::span<void *const> streams, std::size_t capacity)
{
    const auto borrow=original.borrowOriginalNativeContext();
    const auto original_use=borrow.acquire();
    const auto &devices = topology.devices();
    if (isGraphCaptureActive() || !host_device.is_cpu() || streams.size() != devices.size() ||
        authority.worldRank() != topology.worldRank())
        throw std::invalid_argument("Counted allgather requires the admitted original rank, topology and setup streams");
    // Quote every participant before the first claim, including overflow checks.
    for (std::size_t i = 0; i < devices.size(); ++i) (void)topology.memoryFor(i, capacity);
    for (std::size_t i = 0; i < devices.size(); ++i)
        if (!devices[i].is_gpu() || devices[i].type != devices.front().type || reinterpret_cast<std::uintptr_t>(streams[i])<=2 ||
            std::find(devices.begin(), devices.begin() + i, devices[i]) != devices.begin() + i)
            throw std::invalid_argument("Counted allgather membership is not distinct homogeneous local GPUs");
    if(original.degree()!=static_cast<int>(devices.size()) ||
        (original.backend()!=CollectiveBackendType::NCCL && original.backend()!=CollectiveBackendType::RCCL))
        throw std::invalid_argument("Counted allgather requires its original homogeneous native LocalTP context");
    for(std::size_t i=0;i<devices.size();++i) {
        if(original.devices()[i].toLocalDeviceId()!=devices[i])
            throw std::invalid_argument("Counted allgather changed original native communicator order");
        (void)original.nativeCompletionAuthority(static_cast<int>(i));
    }
    auto fabric = std::shared_ptr<DeviceCountedAllGather>(new DeviceCountedAllGather);
    fabric->capacity_ = capacity;
    fabric->devices_.assign(devices.begin(), devices.end());
    fabric->original_=borrow;
    fabric->extents_.resize(devices.size());
    for(std::size_t i=0;i<devices.size();++i)fabric->authorities_.push_back(original.nativeCompletionAuthority(static_cast<int>(i)));
    const auto bytes=NativeCountedAllGatherLayout(devices.size()).bytes();
    for(std::size_t i=0;i<devices.size();++i) {
        auto &extent=fabric->extents_[i];
        extent.claim=authority.claimNewAllocation(devices[i],PhysicalMemoryOwner::ActivationTransportStaging,bytes);
        auto &worker=GPUDeviceContextPool::instance().getContext(devices[i]);
        worker.submitAndWait([&] {
            extent.storage=TransferEngine::instance().allocateDeviceTransferBuffer(bytes,devices[i]);
            auto *backend=getBackendFor(devices[i]);
            // Only cold initialization clears the persistent sticky error and
            // sequence. This exact setup stream also resolves the prep kernel
            // before any graph capture. The setup-only exact-stream join
            // publishes initialization; request reset never repeats it.
            if(!backend || !backend->enqueueNativeCountedAllGatherFrame(extent.storage->mutableDeviceData(),devices.size(),1,capacity,
                    NativeCountedAllGatherPreparation::Initialize,devices[i].ordinal,streams[i]) ||
                !worker.synchronizeStreamChecked(streams[i]))
                throw std::runtime_error("Native counted allgather frame preparation failed");
        });
    }
    return fabric;
}

DeviceCountedAllGatherTransport DeviceCountedAllGather::payloadStorage(
    std::size_t producer, std::size_t consumer) const
{
    if(producer>=devices_.size() || consumer>=devices_.size() || producer==consumer)
        throw std::out_of_range("Counted native allgather edge is foreign or diagonal");
    return DeviceCountedAllGatherTransport::NativeCollective;
}

NativeLocalTPContextUse DeviceCountedAllGather::original() const
{ return original_.acquire(); }

void DeviceCountedAllGather::requireOriginalContext(const ILocalTPContext &expected)const
{
    if(!original_.sameOriginal(expected.borrowOriginalNativeContext()))
        throw std::invalid_argument("Counted allgather installation changed its original native communicator");
    (void)original_.acquire();
}

DeviceCountedAllGatherPacketLayout::DeviceCountedAllGatherPacketLayout(
    std::span<const std::size_t> capacities, std::size_t participant)
    : capacities_(capacities.begin(), capacities.end()), receive_offsets_(capacities.size()), participant_(participant)
{
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (capacities_.size() < 2 || participant_ >= capacities_.size())
        throw std::invalid_argument("Native packet layout requires an original member in a plural cohort");
    for (std::size_t source = 0; source < capacities_.size(); ++source) {
        const auto bytes = capacities_[source];
        if (!bytes || bytes > limit)
            throw std::invalid_argument("Native packet layout requires each source's positive representable capacity");
        maximum_bytes_ = std::max(maximum_bytes_, bytes);
        if (source == participant_) continue;
        if (bytes > limit - received_bytes_)
            throw std::overflow_error("Native packet layout receive geometry overflows");
        receive_offsets_[source] = received_bytes_;
        received_bytes_ += bytes;
    }
}

std::size_t DeviceCountedAllGatherPacketLayout::receiveOffset(std::size_t source) const
{
    if (source == participant_)
        throw std::invalid_argument("Native counted gather stores its own packet in the original local bank");
    return receive_offsets_.at(source);
}

DeviceCountedAllGatherBinding DeviceCountedAllGather::bind(std::size_t participant,
    CapturedTransferMessage message, std::shared_ptr<ITensor> local, std::shared_ptr<ITensor> received)
{
    return bindBanks(participant, message, std::move(local), std::move(received), {});
}

DeviceCountedAllGatherBinding DeviceCountedAllGather::bindPackets(std::uint64_t messageKey,
    std::shared_ptr<const DeviceCountedAllGatherPacketLayout> packets,
    std::shared_ptr<ITensor> local, std::shared_ptr<ITensor> received,
    std::optional<DeviceGenerationCollectiveReceipt> receipt)
{
    if (!packets || packets->degree() != devices_.size())
        throw std::invalid_argument("Native packet layout changed the original counted gather degree");
    const auto participant = packets->participant();
    if (receipt) {
        if (!receipt->isFixedHeadPipelineGather() || receipt->bankBytes() != packets->localBytes())
            throw std::invalid_argument("Native unequal gather requires its exact original terminal send role and packet");
        receipt->requireFor(devices_[participant]);
    }
    const CapturedTransferMessage message{messageKey, packets->maximumBytes()};
    auto binding = bindBanks(participant, message, std::move(local), std::move(received), std::move(packets));
    binding.receipt_ = receipt;
    return binding;
}

DeviceCountedAllGatherBinding DeviceCountedAllGather::bindBanks(std::size_t participant,
    CapturedTransferMessage message, std::shared_ptr<ITensor> local, std::shared_ptr<ITensor> received,
    std::shared_ptr<const DeviceCountedAllGatherPacketLayout> packets)
{
    if (isGraphCaptureActive() || participant >= devices_.size() || !message.fits(capacity_) ||
        !local || !received || local.get() == received.get())
        throw std::invalid_argument("Counted allgather binding requires distinct admitted arena banks and exact message geometry");
    const auto local_bytes = packets ? packets->localBytes() : message.bytes;
    const auto received_bytes = packets ? packets->receivedBytes() : multiply(devices_.size() - 1, message.bytes);
    if (local->size_bytes() < local_bytes || received->size_bytes() < received_bytes)
        throw std::invalid_argument("Counted allgather banks cannot cover their declared physical source packets");
    auto *local_base = dynamic_cast<TensorBase *>(local.get());
    auto *received_base = dynamic_cast<TensorBase *>(received.get());
    if (!local_base || !received_base || local_base->transferStorageOwner() != local_base ||
        received_base->transferStorageOwner() != received_base)
        throw std::invalid_argument("Counted allgather requires canonical arena owners, not offset views");
    const auto begin = reinterpret_cast<std::uintptr_t>(local->gpu_data_ptr());
    const auto destination = reinterpret_cast<std::uintptr_t>(received->gpu_data_ptr());
    constexpr auto limit = std::numeric_limits<std::uintptr_t>::max();
    if (!begin || !destination || begin > limit - local_bytes || destination > limit - received_bytes ||
        (begin < destination + received_bytes && destination < begin + local_bytes))
        throw std::invalid_argument("Counted allgather physical packet banks overlap or are unbound");
    DeviceCountedAllGatherBinding binding;
    binding.owner_ = shared_from_this();
    binding.participant_ = participant;
    (void)original();
    binding.local_capture_ = TransferEngine::instance().retainCapturedDeviceStorage(*local_base, devices_[participant], local_bytes);
    binding.received_capture_ = TransferEngine::instance().retainCapturedDeviceStorage(*received_base, devices_[participant], received_bytes);
    binding.local_owner_ = std::move(local);
    binding.received_owner_ = std::move(received);
    binding.message_ = message;
    binding.packets_ = std::move(packets);
    return binding;
}

void DeviceCountedAllGatherBinding::enqueue(void *stream) const
{ enqueueOnStream(stream,nullptr); }

void DeviceCountedAllGatherBinding::enqueueOnStream(void *stream,const AcquiredDeviceTransferInput *fork) const
{
    if(!owner_ || reinterpret_cast<std::uintptr_t>(stream)<=2 || !local_owner_ || !received_owner_)
        throw std::invalid_argument("Native counted allgather requires retained banks and its exact nondefault stream");
    const auto original_use=owner_->original();
    auto &original=original_use.context();
    const auto device=owner_->devices_[participant_];
    auto *backend=getBackendFor(device);
    auto &transfer=TransferEngine::instance();
    if(!local_capture_ || !received_capture_)throw std::logic_error("Counted native exchange lost its original capture leases");
    transfer.requireCapturedDeviceStorage(*local_capture_, *dynamic_cast<TensorBase *>(local_owner_.get()), device,
        packets_ ? packets_->localBytes() : message_.bytes);
    transfer.requireCapturedDeviceStorage(*received_capture_, *dynamic_cast<TensorBase *>(received_owner_.get()), device,
        packets_ ? packets_->receivedBytes() : multiply(owner_->devices_.size() - 1, message_.bytes));
    if(!fork)TransferEngine::requireDeviceInput(local_owner_.get(),device,stream);
    TransferEngine::requireDeviceOutput(received_owner_.get(),device,stream);
    const NativeCountedAllGatherLayout layout(owner_->devices_.size());
    auto *memory=owner_->extents_[participant_].storage->mutableDeviceData();
    unsigned long long *outgoing = nullptr;
    if (receipt_) {
        if (fork) throw std::logic_error("Native unequal terminal gather requires its original participant stream");
        outgoing = receipt_->native(device);
        TransferEngine::requireDeviceInput(receipt_->owner(), device, stream);
        TransferEngine::requireDeviceOutput(receipt_->owner(), device, stream);
    }
    if(!backend || !backend->enqueueNativeCountedAllGatherFrame(memory,layout.degree,message_.key,message_.bytes,
            NativeCountedAllGatherPreparation::Advance,device.ordinal,stream,outgoing)) {
        (void)original.signalNativeCompletionAbort();
        throw std::runtime_error("Native counted allgather invocation preparation failed");
    }
    auto *operands=reinterpret_cast<ncclLlaminarCompletionOperand*>(static_cast<std::byte*>(memory)+layout.operands());
    // Every original participant submits the same root order. Separate native
    // extent and payload plans cannot be enclosed in an outer native group.
    // Root stays in-place in its own packet; no self packet occupies a receive
    // slot. Every other participant consumes exactly that root's live prefix.
    for(std::size_t root=0;root<layout.degree;++root) {
        void *payload = local_owner_->gpu_data_ptr();
        if (root != participant_) {
            const auto slot = root < participant_ ? root : root - 1;
            const auto offset = packets_ ? packets_->receiveOffset(root) : slot * message_.bytes;
            payload = static_cast<std::byte *>(received_owner_->gpu_data_ptr()) + offset;
        }
        const auto capacity = packets_ ? packets_->capacity(root) : message_.bytes;
        const auto bytes = NativeCollectiveBytes::prefix(extent(root), capacity, operands + root);
        if(!original.nativeBroadcastBytesOnStream(payload,payload,bytes,static_cast<int>(root),
            static_cast<int>(participant_),stream,"device_counted_allgather")) {
            (void)original.signalNativeCompletionAbort();
            throw std::runtime_error("Original native counted allgather broadcast failed");
        }
    }
    if(!backend->validateNativeCountedAllGatherFrame(memory,layout.degree,
        owner_->authorities_[participant_].binding(),device.ordinal,stream)) {
        (void)original.signalNativeCompletionAbort();
        throw std::runtime_error("Native counted allgather completion validation failed");
    }
    // The validator precedes every consumer of these bytes on this exact
    // stream. Direct submissions publish that frontier; an acquired auxiliary
    // fork remains private until its enclosing completion-event join publishes.
    if(!fork)TransferEngine::publishDeviceWrite(received_owner_.get(),device,stream);
    if (receipt_) TransferEngine::publishDeviceWrite(receipt_->owner(), device, stream);
}

void DeviceCountedAllGatherBinding::enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const
{
    if(!owner_ || !input.valid() || input.device()!=owner_->devices_[participant_] ||
        input.sourceOwner()!=dynamic_cast<TensorBase*>(local_owner_.get()))
        throw std::invalid_argument("Native counted allgather requires its original packet's acquired fork");
    // The enclosing TransferEngine fork/join owns output publication. No
    // second coherence transition occurs on this acquired auxiliary stream.
    enqueueOnStream(input.consumerStream(),&input);
}

std::uint64_t *DeviceCountedAllGatherBinding::extent(std::size_t source) const
{
    if (!owner_ || source >= owner_->devices_.size()) throw std::out_of_range("Counted allgather source is outside domain");
    return static_cast<std::uint64_t *>(owner_->extents_[participant_].storage->mutableDeviceData(NativeCountedAllGatherLayout(owner_->devices_.size()).extents()+source * sizeof(std::uint64_t)));
}

const std::uint64_t *DeviceCountedAllGather::extentAddress(std::size_t participant, std::size_t source) const
{
    if (participant >= devices_.size() || source >= devices_.size())
        throw std::out_of_range("Counted allgather extent address is outside domain");
    return static_cast<const std::uint64_t *>(extents_[participant].storage->deviceData(NativeCountedAllGatherLayout(devices_.size()).extents()+source * sizeof(std::uint64_t)));
}

const unsigned long long *DeviceCountedAllGather::payloadBytesAddress(std::size_t participant,std::size_t source) const
{
    (void)extentAddress(participant,source);
    return static_cast<const unsigned long long*>(extents_[participant].storage->deviceData(
        NativeCountedAllGatherLayout(devices_.size()).counters()+source*sizeof(unsigned long long)));
}
const ncclLlaminarByteCompletion *DeviceCountedAllGather::completionAddress(std::size_t participant,std::size_t source) const
{
    (void)extentAddress(participant,source);
    return static_cast<const ncclLlaminarByteCompletion*>(extents_[participant].storage->deviceData(
        NativeCountedAllGatherLayout(devices_.size()).receipts()+source*sizeof(ncclLlaminarByteCompletion)));
}
}
