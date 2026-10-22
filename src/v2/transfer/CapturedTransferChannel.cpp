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
#include "CapturedTransferChannelBatch.h"
#include "CapturedTransferKernelPlan.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
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

        /** @brief Reject misaligned count storage or aliasing with payload capacity.
         * @param count Exact device address of a retained uint64 count.
         * @param payload Exact start of the bound device payload.
         * @param capacity Capture-frozen maximum payload extent. */
        void validateExtentAddress(void *count, void *payload, size_t capacity)
        {
            const auto address = reinterpret_cast<std::uintptr_t>(count);
            const auto start = reinterpret_cast<std::uintptr_t>(payload);
            if (address % alignof(std::uint64_t) != 0)
                throw std::invalid_argument("Device-counted transfer extent must be uint64 aligned");
            // Difference comparisons avoid overflowing one-past-end arithmetic.
            if (address >= start ? address - start < capacity : start - address < sizeof(std::uint64_t))
                throw std::invalid_argument("Device-counted transfer extent overlaps its payload capacity");
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

    CapturedTransferBinding TransferEngine::bindDeviceCountedTransfer(
        CapturedTransferBinding binding, std::shared_ptr<const WorkspaceBufferLease> storage, size_t offset) const
    {
        if (isGraphCaptureActive() || !binding.channel_ || !binding.native_.valid() ||
            binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage || !storage)
            throw std::invalid_argument("Device-counted transfer requires setup-time fixed capacity and retained count storage");
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        if (storage->device() != binding.device_ || storage->backend() != endpoint.backend ||
            resolveBackend(binding.device_) != endpoint.backend)
            throw std::invalid_argument("Device-counted transfer extent belongs to a foreign endpoint/backend");
        if (!storage->contains(offset, sizeof(std::uint64_t)))
            throw std::out_of_range("Device-counted transfer extent exceeds its retained workspace");
        auto *address = storage->data(offset);
        validateExtentAddress(address, binding.device_bytes_, binding.native_.message.bytes);
        binding.extent_storage_ = std::move(storage);
        binding.extent_offset_ = offset;
        binding.native_.device_extent = static_cast<std::uint64_t *>(address);
        binding.native_.extent_source = binding.native_.role == CapturedTransferEndpoint::Producer
            ? CapturedTransferExtentSource::ProducerDevice : CapturedTransferExtentSource::ProducerPublication;
        return binding;
    }

    CapturedTransferBinding TransferEngine::bindDeviceCountedTransfer(
        CapturedTransferBinding binding, std::shared_ptr<DeviceTransferBuffer> storage, size_t offset) const
    {
        if (isGraphCaptureActive() || !binding.channel_ || !binding.native_.valid() ||
            binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage || !storage || !storage->isBound())
            throw std::invalid_argument("Device-counted transfer requires setup-time fixed capacity and bound count storage");
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        if (storage->device_ != binding.device_ || storage->backend_ != endpoint.backend ||
            resolveBackend(binding.device_) != endpoint.backend)
            throw std::invalid_argument("Device-counted transfer extent belongs to a foreign endpoint/backend");
        if (!storage->contains(offset, sizeof(std::uint64_t)))
            throw std::out_of_range("Device-counted transfer extent exceeds its retained buffer");
        auto *address = storage->mutableDeviceData(offset);
        validateExtentAddress(address, binding.device_bytes_, binding.native_.message.bytes);
        binding.extent_storage_ = std::move(storage);
        binding.extent_offset_ = offset;
        binding.native_.device_extent = static_cast<std::uint64_t *>(address);
        binding.native_.extent_source = binding.native_.role == CapturedTransferEndpoint::Producer
            ? CapturedTransferExtentSource::ProducerDevice : CapturedTransferExtentSource::ProducerPublication;
        return binding;
    }

    void TransferEngine::enqueueCapturedTransfer(const CapturedTransferBinding &binding, void *stream) const
    {
        enqueueCapturedTransferImpl(binding, stream, nullptr);
    }

    void TransferEngine::enqueueCapturedExchange(
        std::span<const CapturedTransferBinding> producers,
        std::span<const CapturedTransferBinding> consumers,
        CapturedTransferEndpointScheduling scheduling, void *stream) const
    {
        (void)capturedTransferEndpointSchedulingName(scheduling);
        validateCapturedTransferBatch(producers, stream, nullptr);
        validateCapturedTransferBatch(consumers, stream, nullptr);
        if (producers.front().native_.role != CapturedTransferEndpoint::Producer ||
            consumers.front().native_.role != CapturedTransferEndpoint::Consumer ||
            producers.front().device_ != consumers.front().device_)
            throw std::invalid_argument("Captured exchange requires ordered same-device send/receive phases");
        enqueueCapturedExchangeImpl(producers, consumers, scheduling, stream, nullptr);
    }

    void TransferEngine::enqueueCapturedExchangeImpl(
        std::span<const CapturedTransferBinding> producers,
        std::span<const CapturedTransferBinding> consumers,
        CapturedTransferEndpointScheduling scheduling, void *stream,
        const AcquiredDeviceTransferInput *fork) const
    {
        if (scheduling == CapturedTransferEndpointScheduling::Individual)
        {
            for (const auto &binding : producers) enqueueCapturedTransferImpl(binding, stream, fork);
            for (const auto &binding : consumers) enqueueCapturedTransferImpl(binding, stream, fork);
        }
        else
        {
            // Scheduling was checked before either phase was recorded. Preserve
            // the same send-before-receive ordering and native failure contract.
            enqueueCapturedTransferBatchImpl(producers, stream, fork);
            enqueueCapturedTransferBatchImpl(consumers, stream, fork);
        }
    }

    void TransferEngine::enqueueForkedCapturedExchange(
        std::span<const CapturedTransferBinding> producers,
        std::span<const CapturedTransferBinding> consumers,
        const AcquiredDeviceTransferInput &input, ITensor *output,
        CapturedTransferEndpointScheduling scheduling) const
    {
        (void)capturedTransferEndpointSchedulingName(scheduling);
        auto *destination = dynamic_cast<TensorBase *>(output);
        if (!input.valid() || producers.empty() || consumers.empty() || !destination ||
            destination->transferStorageOwner() != destination || input.sourceOwner() == destination)
            throw std::invalid_argument("Forked captured exchange requires an acquired input and disjoint canonical output");
        requireDeviceOutput(destination, input.device(), input.consumerStream());
        requireDeviceOutput(input.sourceOwner(), input.device(), input.consumerStream());
        const auto source_address = reinterpret_cast<std::uintptr_t>(input.sourceOwner()->gpu_data_ptr());
        const auto destination_address = reinterpret_cast<std::uintptr_t>(destination->gpu_data_ptr());
        if (source_address <= destination_address
                ? destination_address - source_address < input.sourceOwner()->size_bytes()
                : source_address - destination_address < destination->size_bytes())
            throw std::invalid_argument("Forked captured exchange packet banks physically overlap");
        // Authenticate the complete participant before recording any channel.
        // A caller cannot use a valid fork to authorize a different tensor,
        // foreign device, receive-as-send, or unowned workspace publication.
        const auto validate = [&](const auto &bindings, CapturedTransferEndpoint role, TensorBase *owner) {
            for (const auto &binding : bindings)
            {
                const auto *storage = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_);
                if (binding.device_ != input.device() || binding.native_.role != role ||
                    !storage || storage->get() != owner)
                    throw std::invalid_argument("Forked captured exchange endpoint disagrees with its acquired arena owners");
            }
        };
        validate(producers, CapturedTransferEndpoint::Producer, input.sourceOwner());
        validate(consumers, CapturedTransferEndpoint::Consumer, destination);
        // Resolve both phases before the first native operation. A malformed
        // receive cannot leave an otherwise valid producer batch in flight.
        validateCapturedTransferBatch(producers, input.consumerStream(), &input);
        validateCapturedTransferBatch(consumers, input.consumerStream(), &input);
        enqueueCapturedExchangeImpl(producers, consumers, scheduling, input.consumerStream(), &input);
    }

    CapturedTransferChannelBatchEntry TransferEngine::resolveCapturedTransferEndpoint(
        const CapturedTransferBinding &binding, void *stream,
        const AcquiredDeviceTransferInput *fork) const
    {
        if (!stream || !binding.channel_ || !binding.native_.valid())
            throw std::invalid_argument("Captured transfer submission needs a frozen binding and explicit stream");
        const auto &endpoint = binding.channel_->endpoints_[endpointIndex(binding.native_.role)];
        auto *backend = resolveBackend(binding.device_);
        if (!backend || backend != endpoint.backend)
            throw std::logic_error("Captured transfer backend identity changed after binding");
        if (binding.native_.extent_source != CapturedTransferExtentSource::FixedMessage)
        {
            const bool retained = std::visit([&](const auto &storage) {
                using Owner = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<Owner, std::monostate>) return false;
                else if constexpr (std::is_same_v<Owner, std::shared_ptr<DeviceTransferBuffer>>)
                    return storage && storage->device_ == binding.device_ && storage->backend_ == backend &&
                        storage->contains(binding.extent_offset_, sizeof(std::uint64_t)) &&
                        storage->mutableDeviceData(binding.extent_offset_) == binding.native_.device_extent;
                else
                    return storage && storage->device() == binding.device_ && storage->backend() == backend &&
                        storage->contains(binding.extent_offset_, sizeof(std::uint64_t)) &&
                        storage->data(binding.extent_offset_) == binding.native_.device_extent;
            }, binding.extent_storage_);
            if (!retained) throw std::logic_error("Device-counted transfer lost its retained extent owner");
        }
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
            if (binding.native_.role == CapturedTransferEndpoint::Producer && !fork)
                requireDeviceInput(tensor, binding.device_, stream);
            else
                requireDeviceOutput(tensor, binding.device_, stream);
            if (binding.offset_ > tensor->size_bytes() || binding.native_.message.bytes > tensor->size_bytes() - binding.offset_ ||
                static_cast<unsigned char *>(tensor->gpu_data_ptr()) + binding.offset_ != binding.device_bytes_)
                throw std::logic_error("Captured transfer tensor storage changed after binding");
        }
        // Setup certified initialization. Submit one complete typed operation;
        // the backend may fuse nodes without splitting ownership or making
        // TransferEngine dictate a needlessly fragmented native graph.
        auto *mapped_bytes = binding.channel_->mapped_->deviceAlias(binding.device_, CapturedTransferChannel::payload_offset);
        const bool outbound = binding.native_.role == CapturedTransferEndpoint::Producer;
        void *destination = outbound ? mapped_bytes : binding.device_bytes_;
        const void *source = outbound ? binding.device_bytes_ : mapped_bytes;
        return {.binding = binding.native_, .destination = destination, .source = source};
    }

    void TransferEngine::enqueueCapturedTransferImpl(const CapturedTransferBinding &binding,
        void *stream, const AcquiredDeviceTransferInput *fork) const
    {
        const auto entry = resolveCapturedTransferEndpoint(binding, stream, fork);
        auto *const backend = resolveBackend(binding.device_);
        if (!backend->enqueueCapturedTransferChannel(entry.binding, entry.destination, entry.source,
                binding.device_.ordinal, stream))
            throw std::runtime_error("Captured transfer submission failed");
        if (const auto *storage = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_))
            if (binding.native_.role == CapturedTransferEndpoint::Consumer && !fork)
                publishDeviceWrite(storage->get(), binding.device_, stream);
    }

    void TransferEngine::enqueueCapturedTransferBatch(
        std::span<const CapturedTransferBinding> bindings, void *stream) const
    {
        enqueueCapturedTransferBatchImpl(bindings, stream, nullptr);
    }

    void TransferEngine::validateCapturedTransferBatch(
        std::span<const CapturedTransferBinding> bindings, void *stream,
        const AcquiredDeviceTransferInput *fork) const
    {
        if (bindings.empty() || !stream)
            throw std::invalid_argument("Captured endpoint batch requires nonempty original bindings and exact stream");
        const auto device = bindings.front().device_;
        const auto role = bindings.front().native_.role;
        // Full validation precedes submission, including across packet boundaries.
        // A receive bank cannot use a zero current count to disguise aliasing.
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto &binding = bindings[i];
            if (binding.device_ != device || binding.native_.role != role)
                throw std::invalid_argument("Captured endpoint batch mixes physical devices or endpoint roles");
            const auto current = resolveCapturedTransferEndpoint(binding, stream, fork);
            const auto begin = reinterpret_cast<std::uintptr_t>(current.destination);
            for (size_t j = 0; j < i; ++j)
            {
                const auto &prior = bindings[j];
                if (binding.native_.cursor == prior.native_.cursor || binding.native_.control == prior.native_.control)
                    throw std::invalid_argument("Captured endpoint batch duplicates an original channel");
                if (role == CapturedTransferEndpoint::Consumer)
                {
                    const auto previous = reinterpret_cast<std::uintptr_t>(prior.device_bytes_);
                    if (begin <= previous ? previous - begin < binding.native_.message.bytes
                        : begin - previous < prior.native_.message.bytes)
                        throw std::invalid_argument("Captured endpoint batch has overlapping receive capacities");
                    // Publication counts are mutable outputs, unlike shared
                    // producer count inputs. Authenticate their complete byte
                    // ranges against every other endpoint before any enqueue.
                    const auto count = reinterpret_cast<std::uintptr_t>(binding.native_.device_extent);
                    const auto previous_count = reinterpret_cast<std::uintptr_t>(prior.native_.device_extent);
                    const auto overlaps = [](std::uintptr_t first, size_t first_bytes,
                                             std::uintptr_t second, size_t second_bytes) {
                        return first <= second ? second - first < first_bytes : first - second < second_bytes;
                    };
                    constexpr size_t count_bytes = sizeof(std::uint64_t);
                    if ((count && previous_count && overlaps(count, count_bytes, previous_count, count_bytes)) ||
                        (count && overlaps(count, count_bytes, previous, prior.native_.message.bytes)) ||
                        (previous_count && overlaps(previous_count, count_bytes, begin, binding.native_.message.bytes)))
                        throw std::invalid_argument("Captured endpoint batch count publication overlaps another count or payload capacity");
                }
            }
        }
    }

    void TransferEngine::enqueueCapturedTransferBatchImpl(
        std::span<const CapturedTransferBinding> bindings, void *stream,
        const AcquiredDeviceTransferInput *fork) const
    {
        validateCapturedTransferBatch(bindings, stream, fork);
        const auto device = bindings.front().device_;
        const auto role = bindings.front().native_.role;
        auto *const backend = resolveBackend(device);
        for (size_t begin = 0; begin < bindings.size(); begin += CapturedTransferChannelBatch::kCapacity)
        {
            CapturedTransferChannelBatch packet;
            packet.count = static_cast<unsigned>(std::min(bindings.size() - begin,
                size_t{CapturedTransferChannelBatch::kCapacity}));
            bool single = true;
            for (unsigned i = 0; i < packet.count; ++i)
            {
                packet.entries[i] = resolveCapturedTransferEndpoint(bindings[begin + i], stream, fork);
                single = single && capturedTransferBatchKernelPlan(packet.entries[i].binding.message.bytes) ==
                    CapturedTransferKernelPlan::SingleBlock;
            }
            if (!backend->enqueueCapturedTransferChannelBatch(packet, device.ordinal, stream))
                throw std::runtime_error("Captured endpoint batch native submission failed; partial submission cannot be retried");
            // The native graph owns argument copies. Original retained bindings
            // still own all aliases/cursors; no new lifetime or PMA charge exists.
            for (unsigned i = 0; i < packet.count; ++i)
            {
                const auto &binding = bindings[begin + i];
                if (const auto *storage = std::get_if<std::shared_ptr<TensorBase>>(&binding.storage_))
                    if (role == CapturedTransferEndpoint::Consumer && !fork)
                        publishDeviceWrite(storage->get(), device, stream);
            }
            if (PerfStatsCollector::isDomainEnabled("transfer"))
                PerfStatsCollector::addCounter("transfer", "captured_endpoint_batch_submissions", 1.0,
                    {}, device.toString(), {{"endpoints", std::to_string(packet.count)},
                        {"role", role == CapturedTransferEndpoint::Producer ? "producer" : "consumer"},
                        {"lowering", single ? "fused" : "acquire_copy_publish"},
                        {"accounting", "host_submission_or_graph_template"}});
        }
    }
}
