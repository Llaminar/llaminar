/**
 * @file DeviceCountedAllGather.cpp
 * @brief Compose canonical counted TransferEngine messages without host length observation.
 *
 * Setup freezes the rank-local topology and leases. Capture only binds arena
 * addresses and semantic message identity. Replay has no host count readback,
 * new allocation, stream synchronization, or alternate failure path.
 */
#include "DeviceCountedAllGather.h"
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

DeviceCountedAllGatherMemory DeviceCountedAllGather::memoryFor(std::size_t participants, std::size_t capacity)
{
    if (participants < 2) throw std::invalid_argument("Counted allgather needs at least two participants");
    const auto channel = CapturedTransferChannel::memoryFor(capacity);
    const auto cursors = multiply(multiply(2, participants - 1), channel.cursor_bytes_per_device);
    const auto counts = multiply(participants, sizeof(std::uint64_t));
    if (counts > std::numeric_limits<std::size_t>::max() - cursors)
        throw std::overflow_error("Counted allgather extent/cursor geometry overflow");
    return {multiply(participants - 1, channel.mapped_host_bytes), cursors + counts};
}

std::shared_ptr<DeviceCountedAllGather> DeviceCountedAllGather::create(
    PhysicalMemoryAuthority &authority, DeviceId host_device, std::span<const DeviceId> devices,
    std::span<void *const> streams, PeerAccessCoverage coverage, std::size_t capacity)
{
    (void)memoryFor(devices.size(), capacity);
    if (isGraphCaptureActive() || !host_device.is_cpu() || streams.size() != devices.size() ||
        coverage != PeerAccessCoverage::None)
        throw std::invalid_argument("Counted allgather requires setup-time rank-local GPU streams and proven absent P2P");
    for (std::size_t i = 0; i < devices.size(); ++i)
        if (!devices[i].is_gpu() || devices[i].type != devices.front().type || !streams[i] ||
            std::find(devices.begin(), devices.begin() + i, devices[i]) != devices.begin() + i)
            throw std::invalid_argument("Counted allgather membership is not distinct homogeneous local GPUs");
    auto fabric = std::shared_ptr<DeviceCountedAllGather>(new DeviceCountedAllGather);
    fabric->capacity_ = capacity;
    fabric->devices_.assign(devices.begin(), devices.end());
    fabric->extents_.resize(devices.size());
    fabric->channels_.resize(multiply(devices.size(), devices.size()));
    auto &transfer = TransferEngine::instance();
    for (std::size_t i = 0; i < devices.size(); ++i)
    {
        auto &extent = fabric->extents_[i];
        const auto bytes = multiply(devices.size(), sizeof(std::uint64_t));
        extent.claim = authority.claimNewAllocation(devices[i], PhysicalMemoryOwner::ActivationTransportStaging, bytes);
        extent.storage = transfer.allocateDeviceTransferBuffer(bytes, devices[i]);
        // Extents need no memset: local pack writes its count; every remote
        // count is acquired from its producer before import may read it.
        for (std::size_t j = 0; j < devices.size(); ++j)
            if (i != j)
                fabric->channels_[i * devices.size() + j] = transfer.createCapturedTransferChannel(
                    authority, host_device, devices[i], streams[i], devices[j], streams[j], capacity);
    }
    return fabric;
}

DeviceCountedAllGatherBinding DeviceCountedAllGather::bind(std::size_t participant,
    CapturedTransferMessage message, std::shared_ptr<ITensor> local, std::shared_ptr<ITensor> received)
{
    if (isGraphCaptureActive() || participant >= devices_.size() || !message.fits(capacity_) ||
        !local || !received || local.get() == received.get() || local->size_bytes() < message.bytes ||
        received->size_bytes() < multiply(devices_.size() - 1, message.bytes))
        throw std::invalid_argument("Counted allgather binding requires distinct admitted arena banks and exact message geometry");
    auto *local_base = dynamic_cast<TensorBase *>(local.get());
    auto *received_base = dynamic_cast<TensorBase *>(received.get());
    if (!local_base || !received_base || local_base->transferStorageOwner() != local_base ||
        received_base->transferStorageOwner() != received_base)
        throw std::invalid_argument("Counted allgather requires canonical arena owners, not offset views");
    const auto begin = reinterpret_cast<std::uintptr_t>(local->gpu_data_ptr());
    const auto destination = reinterpret_cast<std::uintptr_t>(received->gpu_data_ptr());
    const auto received_bytes = multiply(devices_.size() - 1, message.bytes);
    constexpr auto limit = std::numeric_limits<std::uintptr_t>::max();
    if (!begin || !destination || begin > limit - message.bytes || destination > limit - received_bytes ||
        (begin < destination + received_bytes && destination < begin + message.bytes))
        throw std::invalid_argument("Counted allgather physical packet banks overlap or are unbound");
    DeviceCountedAllGatherBinding binding;
    binding.owner_ = shared_from_this();
    binding.participant_ = participant;
    binding.received_ = received.get();
    auto &transfer = TransferEngine::instance();
    for (std::size_t peer = 0; peer < devices_.size(); ++peer)
    {
        if (peer == participant) continue;
        binding.sends_.push_back(transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(
            channels_[participant * devices_.size() + peer], CapturedTransferEndpoint::Producer, message, local),
            extents_[participant].storage, participant * sizeof(std::uint64_t)));
        const auto slot = peer < participant ? peer : peer - 1;
        binding.receives_.push_back(transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(
            channels_[peer * devices_.size() + participant], CapturedTransferEndpoint::Consumer,
            message, received, slot * message.bytes), extents_[participant].storage, peer * sizeof(std::uint64_t)));
    }
    return binding;
}

void DeviceCountedAllGatherBinding::enqueue(void *stream) const
{
    if (!owner_ || !stream || sends_.empty() || sends_.size() != receives_.size())
        throw std::invalid_argument("Counted allgather requires a complete binding and non-null stream");
    auto &transfer = TransferEngine::instance();
    for (const auto &send : sends_) transfer.enqueueCapturedTransfer(send, stream);
    for (const auto &receive : receives_) transfer.enqueueCapturedTransfer(receive, stream);
}

void DeviceCountedAllGatherBinding::enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const
{
    if (!owner_ || sends_.empty() || sends_.size() != receives_.size() ||
        !input.valid() || input.device() != owner_->devices_[participant_])
        throw std::invalid_argument("Counted allgather requires its exact participant's acquired fork");
    TransferEngine::instance().enqueueForkedCapturedExchange(sends_, receives_, input, received_);
}

std::uint64_t *DeviceCountedAllGatherBinding::extent(std::size_t source) const
{
    if (!owner_ || source >= owner_->devices_.size()) throw std::out_of_range("Counted allgather source is outside domain");
    return static_cast<std::uint64_t *>(owner_->extents_[participant_].storage->mutableDeviceData(source * sizeof(std::uint64_t)));
}

const std::uint64_t *DeviceCountedAllGather::extentAddress(std::size_t participant, std::size_t source) const
{
    if (participant >= devices_.size() || source >= devices_.size())
        throw std::out_of_range("Counted allgather extent address is outside domain");
    return static_cast<const std::uint64_t *>(extents_[participant].storage->deviceData(source * sizeof(std::uint64_t)));
}
}
