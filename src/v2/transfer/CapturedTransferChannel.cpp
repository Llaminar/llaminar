/**
 * @file CapturedTransferChannel.cpp
 * @brief TransferEngine lowering of admitted rank-local captured message channels.
 *
 * Setup claims all three physical allocations through PMA, resolves kernels,
 * then joins the exact cursor-initialization events once before returning ready.
 * Live submission only validates frozen ownership and records acquire, parallel
 * byte copy and release. No live device epoch is read or reset by this host code.
 */
#include "CapturedTransferChannel.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include "tensors/TensorClasses.h"

#include <atomic>
#include <chrono>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>

namespace llaminar2
{
    namespace
    {
        /** @return A never-reused allocation identity, not a request/generation clock. */
        std::uint64_t nextChannelIdentity()
        {
            static std::atomic<std::uint64_t> next{1};
            auto value = next.load(std::memory_order_relaxed);
            do
            {
                if (value == kCapturedTransferAbortEpoch)
                    throw std::overflow_error("Captured transfer allocation identity exhausted");
            } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
            return value;
        }

        /** @return One validated endpoint index; unknown enum values fail before native calls. */
        size_t endpointIndex(CapturedTransferEndpoint role)
        {
            switch (role)
            {
            case CapturedTransferEndpoint::Producer: return 0;
            case CapturedTransferEndpoint::Consumer: return 1;
            }
            throw std::invalid_argument("Captured transfer has an invalid endpoint role");
        }
    }

    CapturedTransferChannelMemory CapturedTransferChannel::memoryFor(size_t capacity)
    {
        if (!capacity) throw std::invalid_argument("Captured transfer needs a positive payload capacity");
        if (capacity > std::numeric_limits<size_t>::max() - payload_offset)
            throw std::overflow_error("Captured transfer payload/control size overflow");
        return {TransferEngine::mappedHostRegionAllocationBytes(payload_offset + capacity),
            sizeof(CapturedTransferCursor)};
    }

    CapturedTransferChannel::~CapturedTransferChannel()
    {
        // Event destruction does not join GPU work. The enclosing graph owner
        // performs its normal terminal join before dropping captured bindings.
        for (auto &endpoint : endpoints_)
            if (endpoint.initialized_event)
                endpoint.backend->destroyEvent(endpoint.initialized_event, endpoint.device.ordinal);
    }

    DeviceId CapturedTransferChannel::endpointDevice(CapturedTransferEndpoint role) const
    { return endpoints_[endpointIndex(role)].device; }

    std::shared_ptr<CapturedTransferChannel> TransferEngine::createCapturedTransferChannel(
        PhysicalMemoryAuthority &authority, DeviceId host_device,
        DeviceId producer, void *producer_setup_stream,
        DeviceId consumer, void *consumer_setup_stream, size_t capacity) const
    {
        if (isGraphCaptureActive() || !host_device.is_cpu() || !producer.is_gpu() || !consumer.is_gpu() || producer == consumer ||
            !producer_setup_stream || !consumer_setup_stream)
            throw std::invalid_argument("Captured transfer requires CPU backing, distinct local GPUs and explicit setup streams");
        const auto geometry = CapturedTransferChannel::memoryFor(capacity);
        auto channel = std::shared_ptr<CapturedTransferChannel>(new CapturedTransferChannel);
        channel->identity_ = {nextChannelIdentity(), capacity};
        constexpr auto owner = PhysicalMemoryOwner::ActivationTransportStaging;
        const std::array devices{producer, consumer};
        const std::array streams{producer_setup_stream, consumer_setup_stream};

        // Claim the complete BOM before materializing any allocation. Partial
        // setup unwinds storage first and each canonical lease second.
        channel->mapped_lease_ = authority.claimNewAllocation(host_device, owner, geometry.mapped_host_bytes);
        for (size_t i = 0; i < devices.size(); ++i)
        {
            auto &endpoint = channel->endpoints_[i];
            endpoint.device = devices[i];
            endpoint.backend = resolveBackend(devices[i]);
            endpoint.lease = authority.claimNewAllocation(devices[i], owner, geometry.cursor_bytes_per_device);
            if (!endpoint.backend || !endpoint.backend->prepareCapturedTransferChannelKernels(devices[i].ordinal,
                    collective_timeout_policy::kDefaultCollectiveTimeoutMs, &endpoint.timeout_ticks) || !endpoint.timeout_ticks)
                throw std::runtime_error("Captured transfer kernel preparation failed on " + devices[i].toString());
        }
        channel->mapped_ = allocateMappedHostRegion(geometry.mapped_host_bytes, devices);
        // The first-touch allocator zeroes every page; placement construction
        // publishes only the immutable identity before either GPU can run.
        auto *control = ::new (channel->mapped_->mutableHostData()) CapturedTransferChannelControl{};
        control->identity = channel->identity_;
        for (size_t i = 0; i < devices.size(); ++i)
        {
            auto &endpoint = channel->endpoints_[i];
            endpoint.cursor = allocateDeviceTransferBuffer(geometry.cursor_bytes_per_device, devices[i]);
            endpoint.initialized_event = endpoint.backend->createEvent(devices[i].ordinal);
            if (!endpoint.initialized_event ||
                !endpoint.backend->memset(endpoint.cursor->mutableDeviceData(), 0, geometry.cursor_bytes_per_device,
                    devices[i].ordinal, streams[i]) ||
                !endpoint.backend->recordEvent(endpoint.initialized_event, devices[i].ordinal, streams[i]))
                throw std::runtime_error("Captured transfer initialization failed on " + devices[i].toString());
        }
        // Creation is a setup-only ready boundary. Joining these exact memset
        // events once makes initialized storage usable by any later capture or
        // cloned parent stream. Importing an uncaptured event inside a captured
        // graph is invalid on native runtimes; retaining a host shadow/reset
        // protocol would add another authority. Neither is needed here.
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        for (const auto &endpoint : channel->endpoints_)
        {
            bool ready = false;
            while (!ready)
            {
                if (!endpoint.backend->queryEvent(endpoint.initialized_event, endpoint.device.ordinal, &ready))
                    throw std::runtime_error("Captured transfer setup event query failed on " + endpoint.device.toString());
                if (!ready)
                {
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("Captured transfer setup deadline expired on " + endpoint.device.toString());
                    std::this_thread::yield();
                }
            }
        }
        return channel;
    }

    CapturedTransferBinding TransferEngine::bindCapturedTransfer(
        std::shared_ptr<CapturedTransferChannel> channel, CapturedTransferEndpoint role,
        CapturedTransferMessage message, std::shared_ptr<DeviceTransferBuffer> storage, size_t offset) const
    {
        const size_t index = endpointIndex(role);
        if (!channel || !message.fits(channel->identity_.capacity) || !storage || !storage->isBound())
            throw std::invalid_argument("Captured transfer binding needs admitted channel, exact message and bound storage");
        const auto &endpoint = channel->endpoints_[index];
        if (storage->device_ != endpoint.device || storage->backend_ != endpoint.backend ||
            resolveBackend(endpoint.device) != endpoint.backend)
            throw std::invalid_argument("Captured transfer buffer/backend does not own the selected endpoint");
        if (!storage->contains(offset, message.bytes))
            throw std::out_of_range("Captured transfer message exceeds device storage");
        CapturedTransferBinding binding;
        binding.channel_ = std::move(channel);
        binding.device_ = endpoint.device;
        binding.device_bytes_ = storage->mutableDeviceData(offset);
        binding.offset_ = offset;
        binding.storage_ = std::move(storage);
        binding.native_ = {
            .control = static_cast<CapturedTransferChannelControl *>(binding.channel_->mapped_->deviceAlias(endpoint.device)),
            .cursor = static_cast<CapturedTransferCursor *>(endpoint.cursor->mutableDeviceData()),
            .expected = binding.channel_->identity_, .message = message, .role = role,
            .timeout_ticks = endpoint.timeout_ticks};
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindCapturedTransfer(
        std::shared_ptr<CapturedTransferChannel> channel, CapturedTransferEndpoint role,
        CapturedTransferMessage message, std::shared_ptr<ITensor> tensor, size_t offset) const
    {
        const size_t index = endpointIndex(role);
        auto *view = dynamic_cast<TensorBase *>(tensor.get());
        auto *base = view ? view->transferStorageOwner() : nullptr;
        if (!channel || !message.fits(channel->identity_.capacity) || !base || base->transferStorageOwner() != base)
            throw std::invalid_argument("Captured tensor transfer needs a canonical physical owner and exact message");
        const auto &endpoint = channel->endpoints_[index];
        std::lock_guard lock(base->coherence_mutex_);
        if (!base->gpu_data_ptr_ || !base->gpu_device_ || *base->gpu_device_ != endpoint.device ||
            resolveBackend(endpoint.device) != endpoint.backend || base->resolveBackend(endpoint.device) != endpoint.backend)
            throw std::invalid_argument("Captured tensor transfer needs existing storage on the exact endpoint");
        if (offset > base->size_bytes() || message.bytes > base->size_bytes() - offset)
            throw std::out_of_range("Captured tensor transfer exceeds physical storage");
        CapturedTransferBinding binding;
        binding.channel_ = std::move(channel);
        binding.device_ = endpoint.device;
        binding.device_bytes_ = static_cast<unsigned char *>(base->gpu_data_ptr_) + offset;
        binding.offset_ = offset;
        // Alias the supplied lifetime instead of requiring a second ownership
        // convention from tensor views which delegate to their physical owner.
        binding.storage_ = std::shared_ptr<TensorBase>(std::move(tensor), base);
        binding.native_ = {
            .control = static_cast<CapturedTransferChannelControl *>(binding.channel_->mapped_->deviceAlias(endpoint.device)),
            .cursor = static_cast<CapturedTransferCursor *>(endpoint.cursor->mutableDeviceData()),
            .expected = binding.channel_->identity_, .message = message, .role = role,
            .timeout_ticks = endpoint.timeout_ticks};
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindCapturedTransfer(
        std::shared_ptr<CapturedTransferChannel> channel, CapturedTransferEndpoint role,
        CapturedTransferMessage message, std::shared_ptr<const WorkspaceBufferLease> storage, size_t offset) const
    {
        const size_t index = endpointIndex(role);
        if (!channel || !message.fits(channel->identity_.capacity) || !storage)
            throw std::invalid_argument("Captured workspace transfer requires an admitted channel and retained region");
        const auto &endpoint = channel->endpoints_[index];
        if (storage->device() != endpoint.device || storage->backend() != endpoint.backend ||
            resolveBackend(endpoint.device) != endpoint.backend)
            throw std::invalid_argument("Captured workspace transfer has foreign endpoint/backend ownership");
        if (!storage->contains(offset, message.bytes))
            throw std::out_of_range("Captured workspace message exceeds its leased region");
        CapturedTransferBinding binding;
        binding.channel_ = std::move(channel);
        binding.device_ = endpoint.device;
        binding.device_bytes_ = storage->data(offset);
        binding.offset_ = offset;
        binding.storage_ = std::move(storage);
        binding.native_ = {
            .control = static_cast<CapturedTransferChannelControl *>(binding.channel_->mapped_->deviceAlias(endpoint.device)),
            .cursor = static_cast<CapturedTransferCursor *>(endpoint.cursor->mutableDeviceData()),
            .expected = binding.channel_->identity_, .message = message, .role = role,
            .timeout_ticks = endpoint.timeout_ticks};
        return binding;
    }

    void TransferEngine::enqueueCapturedTransfer(const CapturedTransferBinding &binding, void *stream) const
    {
        if (!stream || !binding.channel_ || !binding.native_.valid())
            throw std::invalid_argument("Captured transfer submission needs a frozen binding and explicit stream");
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        auto *backend = resolveBackend(binding.device_);
        if (!backend || backend != endpoint.backend)
            throw std::logic_error("Captured transfer backend identity changed after binding");
        TensorBase *tensor = nullptr;
        if (const auto *storage = std::get_if<std::shared_ptr<DeviceTransferBuffer>>(&binding.storage_))
        {
            if (!*storage || (*storage)->device_ != binding.device_ || (*storage)->backend_ != backend ||
                !(*storage)->contains(binding.offset_, binding.native_.message.bytes) ||
                (*storage)->mutableDeviceData(binding.offset_) != binding.device_bytes_)
                throw std::logic_error("Captured transfer scratch binding is stale");
        }
        else if (const auto *storage = std::get_if<std::shared_ptr<const WorkspaceBufferLease>>(&binding.storage_))
        {
            // The lease carries the original block, not the current workspace
            // name map. Growth/retirement cannot redirect this captured address.
            if (!*storage || (*storage)->device() != binding.device_ || (*storage)->backend() != backend ||
                !(*storage)->contains(binding.offset_, binding.native_.message.bytes) ||
                (*storage)->data(binding.offset_) != binding.device_bytes_)
                throw std::logic_error("Captured workspace transfer lost its retained physical owner");
        }
        else
        {
            tensor = std::get<std::shared_ptr<TensorBase>>(binding.storage_).get();
            if (!tensor || tensor->transferStorageOwner() != tensor || tensor->resolveBackend(binding.device_) != backend)
                throw std::logic_error("Captured transfer tensor owner changed");
            if (binding.native_.role == CapturedTransferEndpoint::Producer)
                requireDeviceInput(tensor, binding.device_, stream);
            else
                requireDeviceOutput(tensor, binding.device_, stream);
            if (binding.offset_ > tensor->size_bytes() || binding.native_.message.bytes > tensor->size_bytes() - binding.offset_ ||
                static_cast<unsigned char *>(tensor->gpu_data_ptr()) + binding.offset_ != binding.device_bytes_)
                throw std::logic_error("Captured transfer tensor storage changed after binding");
        }
        // Setup already certified cursor initialization. The live graph has
        // exactly acquire/copy/publish; it cannot import an uncaptured event.
        if (!backend->enqueueCapturedTransferChannelBoundary(binding.native_, CapturedTransferBoundaryOperation::Acquire,
                binding.device_.ordinal, stream))
            throw std::runtime_error("Captured transfer acquire submission failed");
        auto *mapped_bytes = binding.channel_->mapped_->deviceAlias(binding.device_, CapturedTransferChannel::payload_offset);
        const bool outbound = binding.native_.role == CapturedTransferEndpoint::Producer;
        if (!backend->copyDeviceVisibleRegionByKernelOnStream(outbound ? mapped_bytes : binding.device_bytes_,
                outbound ? binding.device_bytes_ : mapped_bytes, binding.native_.message.bytes, binding.device_.ordinal, stream) ||
            !backend->enqueueCapturedTransferChannelBoundary(binding.native_, CapturedTransferBoundaryOperation::Publish,
                binding.device_.ordinal, stream))
            throw std::runtime_error("Captured transfer copy/publication submission failed");
        if (tensor && !outbound) publishDeviceWrite(tensor, binding.device_, stream);
    }
}
