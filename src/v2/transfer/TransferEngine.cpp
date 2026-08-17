/**
 * @file TransferEngine.cpp
 * @brief Canonical event-ordered tensor movement and coherence publication.
 *
 * TransferEngine owns the relationship between byte movement, tensor
 * authority, and producer/consumer ordering. GPU copies are submitted on one
 * exact non-null stream and publish an exact event. Device consumers import
 * that event without blocking the host; host consumers wait only for the event
 * that makes their destination bytes observable. Backend copy functions remain
 * low-level submission mechanisms and never decide tensor coherence.
 */

#include "transfer/TransferEngine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "collective/BackendRouter.h"
#include "collective/ICollectiveBackend.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "utils/DebugEnv.h"
#include "utils/KernelProfiler.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/StackTrace.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Resolve the coherence implementation hidden behind ITensor.
         *
         * Production tensors are TensorBase implementations. Keeping this
         * checked conversion inside TransferEngine preserves ITensor as the
         * orchestration-facing abstraction and turns an unsupported tensor
         * implementation into an immediate contract failure.
         */
        TensorBase *requireTensorBase(ITensor *tensor, const char *operation)
        {
            if (!tensor)
            {
                throw std::invalid_argument(
                    std::string(operation) + " requires a tensor");
            }

            auto *base = dynamic_cast<TensorBase *>(tensor);
            if (!base)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " requires a coherence-aware TensorBase implementation");
            }
            return base;
        }

        /**
         * @brief Resolve the one tensor that owns physical transfer state.
         *
         * Tensor wrappers deliberately delegate pointer and coherence queries
         * to an inner tensor.  TransferEngine must therefore lock and mutate
         * that same inner object, never the wrapper's dormant TensorBase
         * fields.  Requiring a canonical owner at this boundary gives every
         * transfer API the same structural rule.
         */
        TensorBase *requireTransferStorageOwner(
            ITensor *tensor,
            const char *operation)
        {
            TensorBase *base = requireTensorBase(tensor, operation);
            TensorBase *owner = base->transferStorageOwner();
            if (!owner)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " resolved a null transfer-storage owner");
            }
            if (owner->transferStorageOwner() != owner)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " resolved a non-canonical transfer-storage owner");
            }
            return owner;
        }

        /**
         * @brief Resolve the persistent stream owned by a GPU transfer context.
         *
         * Backend APIs deliberately reject null streams. Tensor-aware transfer
         * operations choose their stream here, before touching a backend, so
         * setup, staging, and publication all name one stable owner.
         */
        void *requireTransferStream(
            DeviceId device,
            const char *operation)
        {
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    std::string(operation) +
                    " requires a GPU transfer endpoint");
            }

            void *const stream =
                GPUDeviceContextPool::instance()
                    .getContext(device)
                    .defaultStream();
            if (!stream)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " could not resolve a non-null transfer stream for " +
                    device.toString());
            }
            return stream;
        }

        /**
         * @brief Describe the tensor ownership contract at a failed transfer boundary.
         *
         * A stage can prepare both arena activations and model weights through
         * the same TransferEngine API.  Include the state that controls
         * `uploadFull()` so a hard residency failure identifies the offending
         * tensor and distinguishes an invalid arena publication from a
         * host-resident or prepared-weight classification error.
         */
        std::string tensorTransferState(const TensorBase &tensor)
        {
            std::ostringstream description;
            description
                << " tensor='"
                << (tensor.debugName().empty()
                        ? std::string("(unnamed)")
                        : tensor.debugName())
                << "' coherence=" << to_string(tensor.coherenceState())
                << " residency=" << to_string(tensor.memoryResidency())
                << " prepared_device_state="
                << (tensor.hasPreparedDeviceState() ? "true" : "false")
                << " current_device="
                << (tensor.current_device().has_value()
                        ? tensor.current_device()->toString()
                        : std::string("none"))
                << " gpu_ptr=" << tensor.gpu_data_ptr();
            return description.str();
        }
    } // namespace

    PinnedHostTransferBuffer::PinnedHostTransferBuffer(
        size_t bytes,
        DeviceId registration_device)
        : bytes_(bytes),
          registration_device_(registration_device)
    {
        if (bytes_ == 0 || !registration_device_.is_gpu())
        {
            throw std::invalid_argument(
                "PinnedHostTransferBuffer requires positive bytes and a GPU registration device");
        }
    }

    void PinnedHostTransferBuffer::bind(IBackend *backend)
    {
        if (!backend)
        {
            throw std::invalid_argument(
                "PinnedHostTransferBuffer::bind requires the exact backend");
        }
        if (allocation_)
        {
            if (backend_ == backend)
                return;
            throw std::logic_error(
                "PinnedHostTransferBuffer cannot change backend after binding");
        }

        void *const allocation = backend->allocatePinned(
            bytes_, registration_device_.gpu_ordinal());
        if (!allocation)
        {
            throw std::runtime_error(
                "PinnedHostTransferBuffer allocation failed for " +
                registration_device_.toString() +
                " bytes=" + std::to_string(bytes_));
        }

        /*
         * A captured copy can execute during the transaction that creates the
         * graph.  Deterministic initialization makes any missing protocol
         * publication observable as zeros instead of exposing stale host
         * pages while the packet-level correctness checks diagnose the fault.
         */
        std::memset(allocation, 0, bytes_);
        backend_ = backend;
        allocation_ = allocation;
        ownership_ = Ownership::BackendAllocation;
    }

    void PinnedHostTransferBuffer::bindExternal(
        IBackend *backend,
        void *allocation,
        std::shared_ptr<void> lifetime)
    {
        if (!backend || !allocation || !lifetime)
        {
            throw std::invalid_argument(
                "PinnedHostTransferBuffer::bindExternal requires a backend, address, and lifetime");
        }
        if (allocation_ || ownership_ != Ownership::Unbound)
        {
            throw std::logic_error(
                "PinnedHostTransferBuffer external registration cannot be rebound");
        }
        if (!backend->pinHostMemory(allocation, bytes_))
        {
            throw std::runtime_error(
                "PinnedHostTransferBuffer external registration failed for " +
                registration_device_.toString() +
                " bytes=" + std::to_string(bytes_));
        }
        backend_ = backend;
        allocation_ = allocation;
        ownership_ = Ownership::ExternalRegistration;
        external_lifetime_ = std::move(lifetime);
    }

    PinnedHostTransferBuffer::~PinnedHostTransferBuffer()
    {
        if (allocation_ && backend_ && registration_device_.is_gpu())
        {
            if (ownership_ == Ownership::BackendAllocation)
            {
                backend_->freePinned(
                    allocation_, registration_device_.gpu_ordinal());
            }
            else if (ownership_ == Ownership::ExternalRegistration)
            {
                if (!backend_->unpinHostMemory(allocation_))
                {
                    LOG_ERROR(
                        "PinnedHostTransferBuffer could not unregister external pages for "
                        << registration_device_.toString());
                }
            }
        }
        allocation_ = nullptr;
        bytes_ = 0;
        backend_ = nullptr;
        registration_device_ = DeviceId::invalid();
        ownership_ = Ownership::Unbound;
        external_lifetime_.reset();
    }

    void *PinnedHostTransferBuffer::mutableData(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0))
        {
            throw std::out_of_range(
                "PinnedHostTransferBuffer mutable access requires a bound in-range allocation");
        }
        return static_cast<void *>(
            static_cast<unsigned char *>(allocation_) + offset);
    }

    const void *PinnedHostTransferBuffer::data(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0))
        {
            throw std::out_of_range(
                "PinnedHostTransferBuffer access requires a bound in-range allocation");
        }
        return static_cast<const void *>(
            static_cast<const unsigned char *>(allocation_) + offset);
    }

    DeviceTransferBuffer::DeviceTransferBuffer(
        size_t bytes,
        DeviceId device)
        : bytes_(bytes), device_(device)
    {
        if (bytes_ == 0u || !device_.is_gpu())
        {
            throw std::invalid_argument(
                "DeviceTransferBuffer requires positive bytes and an exact GPU");
        }
    }

    void DeviceTransferBuffer::bind(IBackend *backend)
    {
        if (!backend)
        {
            throw std::invalid_argument(
                "DeviceTransferBuffer::bind requires the exact backend");
        }
        if (allocation_)
        {
            if (backend_ == backend)
                return;
            throw std::logic_error(
                "DeviceTransferBuffer cannot change backend after binding");
        }
        void *const allocation = backend->allocate(
            bytes_, device_.gpu_ordinal());
        if (!allocation)
        {
            throw std::runtime_error(
                "DeviceTransferBuffer allocation failed for " +
                device_.toString() + " bytes=" + std::to_string(bytes_));
        }
        backend_ = backend;
        allocation_ = allocation;
    }

    DeviceTransferBuffer::~DeviceTransferBuffer()
    {
        if (allocation_ && backend_ && device_.is_gpu())
            backend_->free(allocation_, device_.gpu_ordinal());
        allocation_ = nullptr;
        bytes_ = 0u;
        device_ = DeviceId::invalid();
        backend_ = nullptr;
    }

    void *DeviceTransferBuffer::mutableDeviceData(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0u))
        {
            throw std::out_of_range(
                "DeviceTransferBuffer mutable access requires a bound in-range allocation");
        }
        return static_cast<void *>(
            static_cast<unsigned char *>(allocation_) + offset);
    }

    const void *DeviceTransferBuffer::deviceData(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0u))
        {
            throw std::out_of_range(
                "DeviceTransferBuffer access requires a bound in-range allocation");
        }
        return static_cast<const void *>(
            static_cast<const unsigned char *>(allocation_) + offset);
    }

    MappedHostTransferRegion::MappedHostTransferRegion(
        void *allocation,
        size_t bytes,
        std::span<const DeviceId> devices,
        std::shared_ptr<void> lifetime)
        : allocation_(allocation),
          bytes_(bytes),
          devices_(devices.begin(), devices.end()),
          external_lifetime_(std::move(lifetime))
    {
        if (!allocation_ || bytes_ == 0u || devices_.empty() ||
            !external_lifetime_)
        {
            throw std::invalid_argument(
                "MappedHostTransferRegion requires stable pages, positive bytes, endpoints, and a retained lifetime");
        }
        std::sort(devices_.begin(), devices_.end());
        for (std::size_t index = 0; index < devices_.size(); ++index)
        {
            if (!devices_[index].is_valid())
            {
                throw std::invalid_argument(
                    "MappedHostTransferRegion contains an invalid endpoint");
            }
            if (index != 0u && devices_[index] == devices_[index - 1u])
            {
                throw std::invalid_argument(
                    "MappedHostTransferRegion contains a duplicate endpoint " +
                    devices_[index].toString());
            }
        }
        aliases_.reserve(devices_.size());
        registrations_.reserve(2u);
    }

    MappedHostTransferRegion::~MappedHostTransferRegion()
    {
        /*
         * The enclosing graph family proves stream quiescence before releasing
         * this owner. Unregister each runtime while the external mmap lifetime
         * is still held; unmapping first would leave a driver registration
         * pointing at invalid virtual memory.
         */
        for (auto registration = registrations_.rbegin();
             registration != registrations_.rend(); ++registration)
        {
            if (registration->backend &&
                !registration->backend->unregisterExternalMappedHostMemory(
                    allocation_, registration->registration_ordinal))
            {
                LOG_ERROR(
                    "MappedHostTransferRegion could not unregister external pages for backend family "
                    << static_cast<int>(registration->type)
                    << " registration_device="
                    << registration->registration_ordinal);
            }
        }
        bound_ = false;
        aliases_.clear();
        registrations_.clear();
        allocation_ = nullptr;
        bytes_ = 0u;
        devices_.clear();
        external_lifetime_.reset();
    }

    bool MappedHostTransferRegion::isBound() const noexcept
    {
        return bound_ && allocation_ && external_lifetime_ &&
               aliases_.size() == devices_.size();
    }

    void *MappedHostTransferRegion::mutableHostData(size_t offset) const
    {
        if (!isBound() || !contains(offset, 0u))
        {
            throw std::out_of_range(
                "MappedHostTransferRegion host access requires a bound in-range region");
        }
        return static_cast<void *>(
            static_cast<unsigned char *>(allocation_) + offset);
    }

    void *MappedHostTransferRegion::deviceAlias(
        DeviceId device,
        size_t offset) const
    {
        if (!isBound() || !contains(offset, 0u))
        {
            throw std::out_of_range(
                "MappedHostTransferRegion alias access requires a bound in-range region");
        }
        const auto found = std::find_if(
            aliases_.begin(),
            aliases_.end(),
            [&](const DeviceAlias &alias)
            {
                return alias.device == device;
            });
        if (found == aliases_.end() || !found->address)
        {
            throw std::invalid_argument(
                "MappedHostTransferRegion has no alias for " +
                device.toString());
        }
        return static_cast<void *>(
            static_cast<unsigned char *>(found->address) + offset);
    }

    bool MappedHostTransferRegion::hasDevice(DeviceId device) const noexcept
    {
        return isBound() && std::any_of(
            aliases_.begin(),
            aliases_.end(),
            [&](const DeviceAlias &alias)
            {
                return alias.device == device && alias.address != nullptr;
            });
    }

    IBackend *MappedHostTransferRegion::backendFor(
        DeviceId device) const noexcept
    {
        const auto found = std::find_if(
            aliases_.begin(),
            aliases_.end(),
            [&](const DeviceAlias &alias)
            {
                return alias.device == device;
            });
        return found == aliases_.end() ? nullptr : found->backend;
    }

    std::shared_ptr<PinnedHostTransferBuffer>
    TransferEngine::declarePinnedHostBuffer(
        size_t bytes,
        DeviceId registration_device) const
    {
        return std::shared_ptr<PinnedHostTransferBuffer>(
            new PinnedHostTransferBuffer(bytes, registration_device));
    }

    void TransferEngine::bindPinnedHostBuffer(
        PinnedHostTransferBuffer &buffer) const
    {
        IBackend *const backend = resolveBackend(buffer.registration_device_);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::bindPinnedHostBuffer has no backend for " +
                buffer.registration_device_.toString());
        }
        buffer.bind(backend);
    }

    std::shared_ptr<PinnedHostTransferBuffer>
    TransferEngine::allocatePinnedHostBuffer(
        size_t bytes,
        DeviceId registration_device) const
    {
        if (bytes == 0 || !registration_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::allocatePinnedHostBuffer requires positive bytes and a GPU registration device");
        }
        IBackend *const backend = resolveBackend(registration_device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::allocatePinnedHostBuffer has no backend for " +
                registration_device.toString());
        }
        auto buffer = declarePinnedHostBuffer(bytes, registration_device);
        buffer->bind(backend);
        return buffer;
    }

    std::shared_ptr<DeviceTransferBuffer>
    TransferEngine::allocateDeviceTransferBuffer(
        size_t bytes,
        DeviceId device) const
    {
        if (bytes == 0u || !device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::allocateDeviceTransferBuffer requires positive bytes and an exact GPU");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::allocateDeviceTransferBuffer has no backend for " +
                device.toString());
        }
        auto buffer = std::shared_ptr<DeviceTransferBuffer>(
            new DeviceTransferBuffer(bytes, device));
        buffer->bind(backend);
        return buffer;
    }

    std::shared_ptr<PinnedHostTransferBuffer>
    TransferEngine::registerExternalPinnedHostBuffer(
        void *allocation,
        size_t bytes,
        DeviceId registration_device,
        std::shared_ptr<void> lifetime) const
    {
        if (!allocation || bytes == 0 || !registration_device.is_gpu() ||
            !lifetime)
        {
            throw std::invalid_argument(
                "TransferEngine::registerExternalPinnedHostBuffer requires stable pages, positive bytes, a GPU, and a lifetime");
        }
        IBackend *const backend = resolveBackend(registration_device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::registerExternalPinnedHostBuffer has no backend for " +
                registration_device.toString());
        }
        auto buffer = declarePinnedHostBuffer(bytes, registration_device);
        buffer->bindExternal(
            backend, allocation, std::move(lifetime));
        return buffer;
    }

    std::shared_ptr<MappedHostTransferRegion>
    TransferEngine::registerExternalMappedHostRegion(
        void *allocation,
        size_t bytes,
        std::span<const DeviceId> devices,
        std::shared_ptr<void> lifetime) const
    {
        auto region = std::shared_ptr<MappedHostTransferRegion>(
            new MappedHostTransferRegion(
                allocation, bytes, devices, std::move(lifetime)));

        for (const DeviceId device : region->devices_)
        {
            if (device.is_cpu())
            {
                region->aliases_.push_back({
                    .device = device,
                    .address = allocation,
                    .backend = nullptr,
                });
                continue;
            }
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    "TransferEngine mapped region endpoint is neither CPU nor GPU");
            }

            IBackend *const backend = resolveBackend(device);
            if (!backend)
            {
                throw std::runtime_error(
                    "TransferEngine mapped region has no backend for " +
                    device.toString());
            }
            auto registration = std::find_if(
                region->registrations_.begin(),
                region->registrations_.end(),
                [&](const MappedHostTransferRegion::BackendRegistration &item)
                {
                    return item.type == device.type;
                });
            if (registration == region->registrations_.end())
            {
                if (!backend->registerExternalMappedHostMemory(
                        allocation, bytes, device.gpu_ordinal()))
                {
                    throw std::runtime_error(
                        "TransferEngine could not register mapped external pages for " +
                        device.toString());
                }
                region->registrations_.push_back({
                    .type = device.type,
                    .backend = backend,
                    .registration_ordinal = device.gpu_ordinal(),
                });
            }
            else if (registration->backend != backend)
            {
                throw std::logic_error(
                    "TransferEngine resolved multiple backend authorities for one mapped device family");
            }

            void *device_alias = nullptr;
            if (!backend->externalMappedHostDevicePointer(
                    allocation,
                    device.gpu_ordinal(),
                    &device_alias) ||
                !device_alias)
            {
                throw std::runtime_error(
                    "TransferEngine could not resolve mapped external alias for " +
                    device.toString());
            }
            region->aliases_.push_back({
                .device = device,
                .address = device_alias,
                .backend = backend,
            });
        }
        region->bound_ = true;
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            const PerfStatsCollector::Tags tags{
                {"scope", "node_local"},
                {"mapping", "portable_external_host_pages"},
                {"blocking", "false"},
            };
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "mapped_regions_registered",
                1.0,
                "setup",
                "heterogeneous",
                tags);
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "mapped_endpoint_aliases",
                static_cast<double>(region->aliases_.size()),
                "setup",
                "heterogeneous",
                tags);
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "mapped_backend_families",
                static_cast<double>(region->registrations_.size()),
                "setup",
                "heterogeneous",
                tags);
        }
        return region;
    }

    void *TransferEngine::resolveMappedTimelineKernelSignal64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device,
        const char *operation) const
    {
        if (!operation || operation[0] == '\0' || !region.isBound() ||
            !device.is_gpu() || value == 0u ||
            !region.contains(signal_offset, sizeof(std::uint64_t)) ||
            !region.hasDevice(device))
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline kernel binding requires a bound region, aligned signal, positive value, and declared GPU endpoint");
        }
        void *const signal = region.deviceAlias(device, signal_offset);
        if ((reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline kernel signal is not 64-bit aligned");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            !backend->supportsStreamTimelineSignal64(device.gpu_ordinal()))
        {
            throw std::runtime_error(
                std::string("TransferEngine could not bind mapped timeline kernel ") +
                operation + " for " + device.toString());
        }
        return signal;
    }

    MappedTimelineKernelWait64Binding
    TransferEngine::bindMappedTimelineKernelWait64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device) const
    {
        auto *const signal = static_cast<const std::uint64_t *>(
            resolveMappedTimelineKernelSignal64(
                region, signal_offset, value, device, "wait"));
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "device_timeline_kernel_wait_bindings",
                1.0,
                "graph_setup",
                device.toString(),
                {{"scope", "node_local"},
                 {"ordering", "fused_packet_system_acquire"},
                 {"host_blocking", "false"}});
        }
        return MappedTimelineKernelWait64Binding(signal, value, device);
    }

    MappedTimelineKernelPublish64Binding
    TransferEngine::bindMappedTimelineKernelPublish64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device) const
    {
        auto *const signal = static_cast<std::uint64_t *>(
            resolveMappedTimelineKernelSignal64(
                region, signal_offset, value, device, "publication"));
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "device_timeline_kernel_publication_bindings",
                1.0,
                "graph_setup",
                device.toString(),
                {{"scope", "node_local"},
                 {"ordering", "fused_packet_system_release"},
                 {"host_blocking", "false"}});
        }
        return MappedTimelineKernelPublish64Binding(signal, value, device);
    }

    void TransferEngine::enqueueMappedTimelineWait64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device,
        void *stream) const
    {
        if (!region.isBound() || !device.is_gpu() || !stream || value == 0u ||
            !region.contains(signal_offset, sizeof(std::uint64_t)))
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline wait requires a bound region, aligned signal, positive value, GPU, and exact stream");
        }
        void *const signal = region.deviceAlias(device, signal_offset);
        if ((reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline wait signal is not 64-bit aligned");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            !backend->supportsStreamTimelineSignal64(device.gpu_ordinal()) ||
            !backend->streamWaitTimelineSignal64(
                stream,
                signal,
                value,
                device.gpu_ordinal()))
        {
            throw std::runtime_error(
                "TransferEngine could not enqueue mapped 64-bit timeline wait for " +
                device.toString());
        }
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "device_timeline_waits_enqueued",
                1.0,
                "device_epoch",
                device.toString(),
                {
                    {"scope", "node_local"},
                    {"ordering", "exact_stream_64bit_geq"},
                    {"host_blocking", "false"},
                });
        }
    }

    void TransferEngine::enqueueMappedTimelinePublish64(
        const MappedHostTransferRegion &region,
        size_t signal_offset,
        std::uint64_t value,
        DeviceId device,
        void *stream) const
    {
        if (!region.isBound() || !device.is_gpu() || !stream || value == 0u ||
            !region.contains(signal_offset, sizeof(std::uint64_t)))
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline publication requires a bound region, aligned signal, positive value, GPU, and exact stream");
        }
        void *const signal = region.deviceAlias(device, signal_offset);
        if ((reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline publication signal is not 64-bit aligned");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            !backend->supportsStreamTimelineSignal64(device.gpu_ordinal()) ||
            !backend->streamPublishTimelineSignal64(
                stream,
                signal,
                value,
                device.gpu_ordinal()))
        {
            throw std::runtime_error(
                "TransferEngine could not enqueue mapped 64-bit timeline publication for " +
                device.toString());
        }
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "device_timeline_publications_enqueued",
                1.0,
                "device_epoch",
                device.toString(),
                {
                    {"scope", "node_local"},
                    {"ordering", "exact_stream_64bit_fenced"},
                    {"host_blocking", "false"},
                });
        }
    }

    void TransferEngine::enqueueMappedHostToDevice(
        const MappedHostTransferRegion &region,
        size_t source_offset,
        ITensor *destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!destination || !region.isBound() || !device.is_gpu() ||
            !stream || bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueMappedHostToDevice requires a bound region, destination, positive bytes, GPU, and exact stream");
        }
        if (!region.contains(source_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueMappedHostToDevice source region exceeds registered shared pages");
        }

        TensorBase *const destination_owner = requireTransferStorageOwner(
            destination,
            "TransferEngine::enqueueMappedHostToDevice");
        const size_t destination_bytes = destination_owner->size_bytes();
        if (destination_offset > destination_bytes ||
            bytes > destination_bytes - destination_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueueMappedHostToDevice destination region exceeds tensor storage");
        }
        requireDeviceOutput(destination_owner, device, stream);

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueMappedHostToDevice backend identity does not match the mapped registration");
        }
        auto *const destination_ptr =
            static_cast<unsigned char *>(destination_owner->gpu_data_ptr()) +
            destination_offset;
        if (!backend->hostToDeviceOnStream(
                destination_ptr,
                region.mutableHostData(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueMappedHostToDevice failed to enqueue the exact H2D copy");
        }
        publishDeviceWrite(destination_owner, device, stream);
    }

    void TransferEngine::enqueueMappedHostToDevice(
        const MappedHostTransferRegion &region,
        size_t source_offset,
        DeviceTransferBuffer &destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!region.isBound() || !destination.isBound() ||
            !device.is_gpu() || !stream || bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueMappedHostToDevice transport scratch requires bound regions, positive bytes, a GPU, and an exact stream");
        }
        if (!region.contains(source_offset, bytes) ||
            !destination.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueMappedHostToDevice transport scratch exceeds a fixed region");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            backend != destination.backend_ || destination.device_ != device)
        {
            throw std::runtime_error(
                "TransferEngine::enqueueMappedHostToDevice transport scratch backend identity mismatch");
        }
        if (!backend->hostToDeviceOnStream(
                destination.mutableDeviceData(destination_offset),
                region.mutableHostData(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueMappedHostToDevice transport scratch copy enqueue failed");
        }
    }

    void TransferEngine::enqueueDeviceToMappedHost(
        ITensor *source,
        size_t source_offset,
        const MappedHostTransferRegion &region,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source || !region.isBound() || !device.is_gpu() || !stream ||
            bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueDeviceToMappedHost requires a source, bound region, positive bytes, GPU, and exact stream");
        }
        if (!region.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost destination region exceeds registered shared pages");
        }

        TensorBase *const source_owner = requireTransferStorageOwner(
            source,
            "TransferEngine::enqueueDeviceToMappedHost");
        const size_t source_bytes = source_owner->size_bytes();
        if (source_offset > source_bytes || bytes > source_bytes - source_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost source region exceeds tensor storage");
        }
        requireDeviceInput(source_owner, device, stream);

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost backend identity does not match the mapped registration");
        }
        const auto *const source_ptr =
            static_cast<const unsigned char *>(source_owner->gpu_data_ptr()) +
            source_offset;
        if (!backend->deviceToHostOnStream(
                region.mutableHostData(destination_offset),
                source_ptr,
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost failed to enqueue the exact D2H copy");
        }
    }

    DeviceTransferInputFork TransferEngine::recordDeviceInputFork(
        ITensor *source,
        DeviceId device,
        void *producer_stream,
        void *event) const
    {
        if (!source || !device.is_gpu() || !producer_stream || !event)
        {
            throw std::invalid_argument(
                "TransferEngine::recordDeviceInputFork requires a tensor, GPU, exact producer stream, and retained event");
        }

        TensorBase *const source_owner = requireTransferStorageOwner(
            source,
            "TransferEngine::recordDeviceInputFork");
        /* Validate against the canonical graph stream before branching. The
         * dependency ledger must never be asked to reinterpret an auxiliary
         * stream as the tensor's producer. */
        requireDeviceInput(source_owner, device, producer_stream);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        if (!context.recordEventChecked(event, producer_stream))
        {
            throw std::runtime_error(
                "TransferEngine::recordDeviceInputFork could not record the producer frontier");
        }
        return DeviceTransferInputFork(
            source_owner, device, producer_stream, event);
    }

    AcquiredDeviceTransferInput TransferEngine::acquireDeviceInputFork(
        const DeviceTransferInputFork &publication,
        void *consumer_stream) const
    {
        if (!publication.valid() || !consumer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::acquireDeviceInputFork requires a valid publication and exact consumer stream");
        }
        auto &context =
            GPUDeviceContextPool::instance().getContext(publication.device_);
        if (!context.waitEventChecked(publication.event_, consumer_stream))
        {
            throw std::runtime_error(
                "TransferEngine::acquireDeviceInputFork could not enqueue the producer-event wait");
        }
        return AcquiredDeviceTransferInput(
            publication.source_owner_,
            publication.device_,
            consumer_stream,
            publication.event_);
    }

    void TransferEngine::enqueueDeviceToMappedHost(
        const AcquiredDeviceTransferInput &source,
        size_t source_offset,
        const MappedHostTransferRegion &region,
        size_t destination_offset,
        size_t bytes) const
    {
        if (!source.valid() || !region.isBound() || bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueDeviceToMappedHost requires an acquired fork, bound region, and positive byte count");
        }
        if (!region.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost fork destination exceeds registered shared pages");
        }
        const size_t source_bytes = source.source_owner_->size_bytes();
        if (source_offset > source_bytes || bytes > source_bytes - source_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost fork source exceeds tensor storage");
        }
        const auto *const source_ptr =
            static_cast<const unsigned char *>(
                source.source_owner_->gpu_data_ptr()) +
            source_offset;
        IBackend *const backend = resolveBackend(source.device_);
        if (!source_ptr || !backend ||
            backend != region.backendFor(source.device_))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost fork backend or stable tensor storage is incomplete");
        }
        if (!backend->deviceToHostOnStream(
                region.mutableHostData(destination_offset),
                source_ptr,
                bytes,
                source.device_.gpu_ordinal(),
                source.consumer_stream_))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost fork D2H enqueue failed");
        }
    }

    void TransferEngine::enqueueDeviceToMappedHost(
        const DeviceTransferBuffer &source,
        size_t source_offset,
        const MappedHostTransferRegion &region,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source.isBound() || !region.isBound() || !device.is_gpu() ||
            !stream || bytes == 0u)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueDeviceToMappedHost transport scratch requires bound regions, positive bytes, a GPU, and an exact stream");
        }
        if (!source.contains(source_offset, bytes) ||
            !region.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToMappedHost transport scratch exceeds a fixed region");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != region.backendFor(device) ||
            backend != source.backend_ || source.device_ != device)
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost transport scratch backend identity mismatch");
        }
        if (!backend->deviceToHostOnStream(
                region.mutableHostData(destination_offset),
                source.deviceData(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToMappedHost transport scratch copy enqueue failed");
        }
    }

    void TransferEngine::buildMappedTimelineTransaction(
        IGPUGraphCapture &destination,
        std::span<const MappedTimelineTransactionStep> ordered_steps,
        DeviceId device) const
    {
        if (!device.is_gpu() || ordered_steps.empty() ||
            !destination.executionStream() || destination.hasExecutable() ||
            destination.nodeCount() != 0u)
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline transaction requires an empty exact-device graph and non-empty steps");
        }
        IBackend *const backend = resolveBackend(device);
        if (!backend ||
            !backend->supportsStreamTimelineSignal64(device.gpu_ordinal()))
        {
            throw std::invalid_argument(
                "TransferEngine mapped timeline transaction requires native 64-bit timeline support for " +
                device.toString());
        }

        std::vector<GPUOrderedTimelineStep> lowered;
        lowered.reserve(ordered_steps.size());
        const auto requireName = [](const char *name)
        {
            return name && name[0] != '\0';
        };
        for (const auto &step : ordered_steps)
        {
            if (const auto *fragment =
                    std::get_if<MappedTimelineCapturedFragment>(&step))
            {
                if (!requireName(fragment->name) || !fragment->capture ||
                    fragment->capture == &destination ||
                    !fragment->capture->executionStream() ||
                    fragment->capture->nodeCount() == 0u)
                {
                    throw std::invalid_argument(
                        "TransferEngine mapped timeline fragment is incomplete or aliases its destination");
                }
                lowered.push_back({
                    .name = fragment->name,
                    .kind =
                        GPUOrderedTimelineStepKind::CapturedFragment,
                    .capture = fragment->capture,
                    .signal = nullptr,
                    .value = 0u,
                });
                continue;
            }

            const MappedHostTransferRegion *region = nullptr;
            const char *name = nullptr;
            size_t signal_offset = 0u;
            std::uint64_t value = 0u;
            GPUOrderedTimelineStepKind kind =
                GPUOrderedTimelineStepKind::WaitValue64;
            if (const auto *wait =
                    std::get_if<MappedTimelineWait64>(&step))
            {
                region = wait->region;
                name = wait->name;
                signal_offset = wait->signal_offset;
                value = wait->value;
            }
            else
            {
                const auto &publish =
                    std::get<MappedTimelinePublish64>(step);
                region = publish.region;
                name = publish.name;
                signal_offset = publish.signal_offset;
                value = publish.value;
                kind = GPUOrderedTimelineStepKind::PublishValue64;
            }
            if (!requireName(name) || !region || !region->isBound() ||
                !region->hasDevice(device) || value == 0u ||
                !region->contains(signal_offset, sizeof(std::uint64_t)) ||
                region->backendFor(device) != backend)
            {
                throw std::invalid_argument(
                    "TransferEngine mapped timeline step has incomplete region/device/value ownership");
            }
            void *const signal = region->deviceAlias(device, signal_offset);
            if ((reinterpret_cast<std::uintptr_t>(signal) &
                 (alignof(std::uint64_t) - 1u)) != 0u)
            {
                throw std::invalid_argument(
                    "TransferEngine mapped timeline graph signal is not 64-bit aligned");
            }
            lowered.push_back({
                .name = name,
                .kind = kind,
                .capture = nullptr,
                .signal = signal,
                .value = value,
            });
        }
        if (!destination.buildOrderedTimelineTransaction(lowered))
        {
            throw std::runtime_error(
                "TransferEngine could not lower mapped timeline transaction for " +
                device.toString());
        }
        if (PerfStatsCollector::isDomainEnabled(
                "moe_overlay_activation_epoch"))
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_activation_epoch",
                "native_graph_timeline_transactions_built",
                1.0,
                "graph_setup",
                device.toString(),
                {
                    {"host_blocking", "false"},
                    {"ordering", "device_owned_graph_timeline_nodes"},
                    {"steps", std::to_string(lowered.size())},
                });
        }
    }

    void TransferEngine::enqueuePinnedHostToDevice(
        const PinnedHostTransferBuffer &pinned_source,
        size_t source_offset,
        ITensor *destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!destination || !device.is_gpu() || !stream || bytes == 0)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueuePinnedHostToDevice requires a destination, positive bytes, a GPU, and an exact stream");
        }
        if (pinned_source.registration_device_ != device ||
            !pinned_source.contains(source_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueuePinnedHostToDevice source region or registration device is invalid");
        }

        TensorBase *const destination_owner = requireTransferStorageOwner(
            destination,
            "TransferEngine::enqueuePinnedHostToDevice");
        const size_t destination_bytes = destination_owner->size_bytes();
        if (destination_offset > destination_bytes ||
            bytes > destination_bytes - destination_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueuePinnedHostToDevice destination region exceeds tensor storage");
        }
        requireDeviceOutput(destination_owner, device, stream);

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != pinned_source.backend_)
        {
            throw std::runtime_error(
                "TransferEngine::enqueuePinnedHostToDevice backend identity does not match the pinned registration");
        }
        auto *const destination_ptr =
            static_cast<unsigned char *>(destination_owner->gpu_data_ptr()) +
            destination_offset;
        if (!backend->hostToDeviceOnStream(
                destination_ptr,
                pinned_source.data(source_offset),
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueuePinnedHostToDevice failed to enqueue the exact H2D copy");
        }
        publishDeviceWrite(destination_owner, device, stream);
    }

    void TransferEngine::enqueueDeviceToPinnedHost(
        ITensor *source,
        size_t source_offset,
        const PinnedHostTransferBuffer &pinned_destination,
        size_t destination_offset,
        size_t bytes,
        DeviceId device,
        void *stream) const
    {
        if (!source || !device.is_gpu() || !stream || bytes == 0)
        {
            throw std::invalid_argument(
                "TransferEngine::enqueueDeviceToPinnedHost requires a source, positive bytes, a GPU, and an exact stream");
        }
        if (pinned_destination.registration_device_ != device ||
            !pinned_destination.contains(destination_offset, bytes))
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToPinnedHost destination region or registration device is invalid");
        }

        TensorBase *const source_owner = requireTransferStorageOwner(
            source,
            "TransferEngine::enqueueDeviceToPinnedHost");
        const size_t source_bytes = source_owner->size_bytes();
        if (source_offset > source_bytes || bytes > source_bytes - source_offset)
        {
            throw std::out_of_range(
                "TransferEngine::enqueueDeviceToPinnedHost source region exceeds tensor storage");
        }
        requireDeviceInput(source_owner, device, stream);

        IBackend *const backend = resolveBackend(device);
        if (!backend || backend != pinned_destination.backend_)
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToPinnedHost backend identity does not match the pinned registration");
        }
        const auto *const source_ptr =
            static_cast<const unsigned char *>(source_owner->gpu_data_ptr()) +
            source_offset;
        if (!backend->deviceToHostOnStream(
                pinned_destination.mutableData(destination_offset),
                source_ptr,
                bytes,
                device.gpu_ordinal(),
                stream))
        {
            throw std::runtime_error(
                "TransferEngine::enqueueDeviceToPinnedHost failed to enqueue the exact D2H copy");
        }
    }

    // ============================================================================
    // Singleton
    // ============================================================================

    TransferEngine &TransferEngine::instance()
    {
        static TransferEngine engine;
        return engine;
    }

    void TransferEngine::prepareDeviceInput(
        ITensor *tensor,
        DeviceId target_device,
        void *stream)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceInput requires a GPU target");
        }
        if (!stream)
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceInput requires the exact "
                "non-null consumer stream");
        }
        if (isGraphCaptureActive())
        {
            throw std::logic_error(
                "TransferEngine::prepareDeviceInput is forbidden during GPU "
                "graph capture; capture consumers must use requireDeviceInput");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::prepareDeviceInput");
        if (!base->ensureOnDevice(target_device, stream))
        {
            throw std::runtime_error(
                "TransferEngine::prepareDeviceInput failed to place tensor on " +
                target_device.toString());
        }
        if (!base->gpu_data_ptr() ||
            !base->is_on_device(target_device))
        {
            throw std::runtime_error(
                "TransferEngine::prepareDeviceInput completed without storage on " +
                target_device.toString() +
                tensorTransferState(*base));
        }
    }

    void TransferEngine::requireDeviceInput(
        ITensor *tensor,
        DeviceId target_device,
        void *consumer_stream)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceInput requires a GPU target");
        }
        if (!consumer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceInput requires the exact "
                "non-null consumer stream");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::requireDeviceInput");
        std::lock_guard<std::mutex> lock(base->coherence_mutex_);

        if (!base->gpu_data_ptr_ ||
            !base->gpu_device_.has_value() ||
            *base->gpu_device_ != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput requires pre-existing "
                "storage on the exact device " +
                target_device.toString() + tensorTransferState(*base));
        }

        /*
         * An earlier stage in this exact capture transaction has recorded, but
         * not yet executed, the producer kernel.  Stable storage plus the frozen
         * topological ledger is the complete proof here; publishing global device
         * authority or importing an old event would both describe the wrong
         * generation.  Unknown and graph-external inputs continue through the
         * strict coherence/event checks below.
         */
        if (auto *ledger = currentGraphCaptureDependencyLedger())
        {
            const auto disposition = ledger->classifyInput(
                base, target_device, consumer_stream);
            if (disposition ==
                    GraphCaptureDependencyLedger::InputDisposition::InternalRecorded ||
                disposition ==
                    GraphCaptureDependencyLedger::InputDisposition::RetainedParentRecorded ||
                disposition ==
                    GraphCaptureDependencyLedger::InputDisposition::SetupAddressOnlyExternal)
            {
                /*
                 * A retained-parent import is deliberately no more globally
                 * authoritative than an ordinary earlier recorded producer.
                 * Its child graph records a stable pointer read, and the
                 * topology-owned parent later inserts the exact child edge.
                 * A setup-only external is similarly address-authoritative but
                 * not payload-authoritative. It is admitted only when the
                 * ledger names it as a declared arena frontier; transaction
                 * zero performs the live-byte preflight before graph launch.
                 */
                return;
            }
        }

        if (!::llaminar2::isDeviceValid(base->coherence_state_))
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput requires globally valid "
                "external input bytes on " +
                target_device.toString() + tensorTransferState(*base));
        }

        if (!base->device_completion_event_)
            return;

        /*
         * A capture body may consume only dependencies established before
         * beginCapture(). Importing an event recorded by uncaptured work here is
         * rejected by CUDA/HIP and, more importantly, hides an incomplete graph
         * boundary. The exact event/stream identity is established by the
         * executor's pre-capture input pass.
         */
        if (isGraphCaptureActive())
        {
            if (base->last_joined_completion_event_ ==
                    base->device_completion_event_ &&
                base->last_joined_consumer_stream_ == consumer_stream)
            {
                return;
            }

            throw std::runtime_error(
                "TransferEngine::requireDeviceInput found an external producer "
                "event that was not joined to the exact consumer stream before "
                "GPU graph capture began for " +
                target_device.toString() + tensorTransferState(*base));
        }

        if (!base->event_device_.has_value() ||
            *base->event_device_ != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput found completion-event "
                "ownership that does not match " +
                target_device.toString() + tensorTransferState(*base));
        }

        IBackend *backend = base->resolveBackend(target_device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput could not resolve backend "
                "for " +
                target_device.toString() + tensorTransferState(*base));
        }

        const int backend_device_id = target_device.gpu_ordinal();
        if (!backend->streamWaitEvent(
                consumer_stream,
                base->device_completion_event_,
                backend_device_id))
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput failed to join producer "
                "event on consumer stream for " +
                target_device.toString() + tensorTransferState(*base));
        }

        /*
         * Record only after the backend accepts the wait. This is a proof about
         * one event generation on one stream, not a general coherence flag.
         * The next device publication clears it before re-recording the event.
         */
        base->last_joined_completion_event_ = base->device_completion_event_;
        base->last_joined_consumer_stream_ = consumer_stream;
    }

    void TransferEngine::prepareHostInput(ITensor *tensor)
    {
        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::prepareHostInput");
        const TransferResult result = instance().download(base);
        if (!result.success)
        {
            throw std::runtime_error(
                "TransferEngine::prepareHostInput failed to materialize tensor: " +
                result.error);
        }
        if (!base->hostValid())
        {
            throw std::runtime_error(
                "TransferEngine::prepareHostInput completed without valid host storage");
        }
    }

    void TransferEngine::allocateDeviceStorage(
        ITensor *tensor,
        DeviceId target_device)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::allocateDeviceStorage requires a GPU target");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::allocateDeviceStorage");
        if (!base->allocateOnDevice(target_device))
        {
            throw std::runtime_error(
                "TransferEngine::allocateDeviceStorage failed to allocate tensor on " +
                target_device.toString());
        }
        if (!base->gpu_data_ptr() ||
            !base->current_device().has_value() ||
            *base->current_device() != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::allocateDeviceStorage completed without storage on " +
                target_device.toString());
        }
    }

    void TransferEngine::prepareDeviceOutput(
        ITensor *tensor,
        DeviceId target_device,
        void *stream)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceOutput requires the exact "
                "non-null producer stream");
        }
        if (isGraphCaptureActive())
        {
            throw std::logic_error(
                "TransferEngine::prepareDeviceOutput is forbidden during GPU "
                "graph capture; capture writers must use requireDeviceOutput");
        }
        allocateDeviceStorage(tensor, target_device);
    }

    void TransferEngine::requireDeviceOutput(
        ITensor *tensor,
        DeviceId target_device,
        void *producer_stream)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceOutput requires a GPU target");
        }
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceOutput requires the exact "
                "non-null producer stream");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::requireDeviceOutput");
        std::lock_guard<std::mutex> lock(base->coherence_mutex_);
        if (!base->gpu_data_ptr_ ||
            !base->gpu_device_.has_value() ||
            *base->gpu_device_ != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceOutput requires pre-existing "
                "storage on the exact device " +
                target_device.toString() + tensorTransferState(*base));
        }
    }

    void TransferEngine::publishDeviceWrite(
        TensorBase *tensor,
        DeviceId device,
        void *producer_stream)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishDeviceWrite requires a GPU device");
        }
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::publishDeviceWrite requires the exact "
                "non-null producer stream");
        }

        if (auto *ledger = currentGraphCaptureDependencyLedger())
        {
            ledger->recordStagePublication(
                tensor, device, producer_stream);
            std::lock_guard<std::mutex> lock(tensor->coherence_mutex_);
            if (!tensor->gpu_data_ptr_ ||
                !tensor->gpu_device_.has_value() ||
                *tensor->gpu_device_ != device)
            {
                throw std::runtime_error(
                    "TransferEngine::publishDeviceWrite recorded a graph output "
                    "without stable storage on " +
                    device.toString() + tensorTransferState(*tensor));
            }

            /*
             * Recording is not execution.  The graph boundary publishes the
             * actual completed generation after launch, with one exact stream
             * event.  Do not mutate coherence or add per-stage event nodes here.
             */
            return;
        }
        tensor->publishDeviceWriteStateWithEvent(device, producer_stream);
    }

    void TransferEngine::publishDeviceWrite(
        ITensor *tensor,
        DeviceId device,
        void *producer_stream)
    {
        publishDeviceWrite(
            requireTensorBase(tensor, "TransferEngine::publishDeviceWrite"),
            device,
            producer_stream);
    }

    void TransferEngine::publishCompletedDeviceWrite(
        TensorBase *tensor,
        DeviceId device)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishCompletedDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishCompletedDeviceWrite requires a GPU device");
        }
        tensor->publishCompletedDeviceWriteState(device);
    }

    void TransferEngine::publishCompletedDeviceWrite(
        ITensor *tensor,
        DeviceId device)
    {
        publishCompletedDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishCompletedDeviceWrite"),
            device);
    }

    void TransferEngine::publishCurrentDeviceWrite(
        TensorBase *tensor,
        void *producer_stream)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishCurrentDeviceWrite");
        const auto device = tensor->current_device();
        if (!device.has_value() || !device->is_gpu())
        {
            throw std::runtime_error(
                "TransferEngine::publishCurrentDeviceWrite requires an "
                "unambiguous current GPU device");
        }
        publishDeviceWrite(tensor, *device, producer_stream);
    }

    void TransferEngine::publishCurrentDeviceWrite(
        ITensor *tensor,
        void *producer_stream)
    {
        publishCurrentDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishCurrentDeviceWrite"),
            producer_stream);
    }

    void TransferEngine::publishGraphOwnedDeviceWrite(
        TensorBase *tensor,
        DeviceId device)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishGraphOwnedDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishGraphOwnedDeviceWrite requires a GPU device");
        }
        tensor->publishGraphOwnedDeviceWriteState(device);
    }

    void TransferEngine::publishGraphOwnedDeviceWrite(
        ITensor *tensor,
        DeviceId device)
    {
        publishGraphOwnedDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishGraphOwnedDeviceWrite"),
            device);
    }

    void TransferEngine::publishGraphOwnedCurrentDeviceWrite(
        TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishGraphOwnedCurrentDeviceWrite");
        const auto device = tensor->current_device();
        if (!device.has_value() || !device->is_gpu())
        {
            throw std::runtime_error(
                "TransferEngine::publishGraphOwnedCurrentDeviceWrite requires "
                "an unambiguous current GPU device");
        }
        publishGraphOwnedDeviceWrite(tensor, *device);
    }

    void TransferEngine::publishGraphOwnedCurrentDeviceWrite(ITensor *tensor)
    {
        publishGraphOwnedCurrentDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishGraphOwnedCurrentDeviceWrite"));
    }

    void TransferEngine::publishHostWrite(TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishHostWrite");
        tensor->publishHostWriteState();
    }

    void TransferEngine::publishHostWrite(ITensor *tensor)
    {
        publishHostWrite(
            requireTensorBase(tensor, "TransferEngine::publishHostWrite"));
    }

    void TransferEngine::publishSynchronized(TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishSynchronized");
        tensor->publishSynchronizedState();
    }

    void TransferEngine::publishSynchronized(ITensor *tensor)
    {
        publishSynchronized(
            requireTensorBase(tensor, "TransferEngine::publishSynchronized"));
    }

    // ============================================================================
    // planTransfer — pure logic, no side effects
    // ============================================================================

    TransferMethod TransferEngine::planTransfer(DeviceId src, DeviceId dst, MemoryResidency residency)
    {
        // Host-resident tensors never move to device
        if (residency == MemoryResidency::HOST_RESIDENT)
            return TransferMethod::NOOP;

        // Mapped memory is always in-place
        if (residency == MemoryResidency::MAPPED)
            return TransferMethod::MAPPED_NOOP;

        // Same device — nothing to do
        if (src == dst)
            return TransferMethod::NOOP;

        // CPU ↔ GPU
        if (src.is_cpu() && dst.is_gpu())
            return TransferMethod::HOST_TO_DEVICE;

        if (src.is_gpu() && dst.is_cpu())
            return TransferMethod::DEVICE_TO_HOST;

        // GPU ↔ GPU
        if (src.is_gpu() && dst.is_gpu())
        {
            // Same vendor (CUDA↔CUDA or ROCm↔ROCm) — direct P2P
            if (src.type == dst.type)
                return TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND;

            // Cross-vendor (CUDA↔ROCm) — host-staged transfer
            return TransferMethod::HOST_STAGED;
        }

        // CPU → CPU is a no-op (same host)
        if (src.is_cpu() && dst.is_cpu())
            return TransferMethod::NOOP;

        LOG_WARN("[TransferEngine] Unhandled device combination: "
                 << src.toString() << " → " << dst.toString());
        return TransferMethod::NOOP;
    }

    std::string TransferEngine::describeTransferPlan(DeviceId src, DeviceId dst, MemoryResidency residency)
    {
        auto method = planTransfer(src, dst, residency);
        std::ostringstream ss;
        ss << src.toString() << " → " << dst.toString()
           << " [" << to_string(residency) << "] → " << to_string(method);
        return ss.str();
    }

    // ============================================================================
    // execute — dispatch on TransferMethod
    // ============================================================================

    TransferResult TransferEngine::execute(const TransferRequest &request)
    {
        auto start = std::chrono::steady_clock::now();

        TransferResult result;
        switch (request.method)
        {
        case TransferMethod::NOOP:
        case TransferMethod::MAPPED_NOOP:
            result = TransferResult::ok(request.method);
            break;

        case TransferMethod::HOST_TO_DEVICE:
            result = executeHostToDevice(request);
            break;

        case TransferMethod::DEVICE_TO_HOST:
            result = executeDeviceToHost(request);
            break;

        case TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND:
            result = executeDeviceToDeviceSameBackend(request);
            break;

        case TransferMethod::HOST_STAGED:
            result = executeHostStaged(request);
            break;
        }

        auto end = std::chrono::steady_clock::now();
        result.elapsed_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        result.method_used = request.method;

        traceTransfer(request, result);
        return result;
    }

    // ============================================================================
    // High-level TensorBase API
    // ============================================================================

    TransferResult TransferEngine::upload(TensorBase *tensor, DeviceId target_device)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * uploadFull() is the single implementation of allocation, primary
         * device promotion, H2D transfer, event joining, and coherence
         * publication.  Keeping a second implementation here previously let a
         * successful copy land in secondary storage without updating
         * gpu_data_ptr_ or gpu_device_.  Public callers now enter the same
         * lifecycle as TensorBase::ensureOnDevice().
         */
        std::lock_guard<std::mutex> lock(tensor->coherence_mutex_);
        return uploadFull(tensor, target_device, nullptr);
    }

    TransferResult TransferEngine::download(TensorBase *tensor)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * downloadFull() owns completion-event validation, D2H ordering, and
         * host publication.  Delegating to it prevents wrapper and concrete
         * tensors from acquiring subtly different host-read semantics.
         */
        std::lock_guard<std::mutex> lock(tensor->coherence_mutex_);
        return downloadFull(tensor, nullptr);
    }

    TransferResult TransferEngine::transferActivation(
        TensorBase *tensor,
        DeviceId target_device,
        size_t bytes_override)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * Select the actual authoritative source. The authoritative copy may
         * live in a secondary device buffer after a previous pipeline handoff,
         * so gpu_device_ alone is not a sufficient source-of-truth.
         */
        DeviceId src = DeviceId::cpu();
        const auto authoritative_device = tensor->getAuthoritativeDevice();
        if (authoritative_device.has_value() &&
            authoritative_device->is_gpu() &&
            tensor->deviceValid())
        {
            src = *authoritative_device;
        }
        else if (!tensor->hostValid())
        {
            return TransferResult::fail(TransferMethod::NOOP,
                                        "tensor has no valid data on any device");
        }

        const MemoryResidency residency = tensor->memoryResidency();
        TransferMethod method = planTransfer(src, target_device, residency);

        if (method == TransferMethod::NOOP || method == TransferMethod::MAPPED_NOOP)
        {
            /*
             * A valid copy can be stored in the secondary map. Promote it so
             * gpu_data_ptr() and current_device() describe the same storage
             * without pretending that a transfer occurred.
             */
            if (target_device.is_gpu() &&
                (!tensor->gpu_device_.has_value() ||
                 *tensor->gpu_device_ != target_device))
            {
                void *target_ptr =
                    tensor->getOrAllocateDeviceBuffer(target_device);
                if (!target_ptr)
                {
                    return TransferResult::fail(
                        method,
                        "authoritative device has no activation buffer on " +
                            target_device.toString());
                }
                if (tensor->gpu_device_.has_value() && tensor->gpu_data_ptr_)
                {
                    tensor->secondary_device_buffers_
                        [TensorBase::packDeviceId(*tensor->gpu_device_)] =
                        tensor->gpu_data_ptr_;
                }
                tensor->gpu_device_ = target_device;
                tensor->gpu_data_ptr_ = target_ptr;
                tensor->secondary_device_buffers_.erase(
                    TensorBase::packDeviceId(target_device));
            }
            return TransferResult::ok(method);
        }

        TransferRequest req;
        req.source = makeMemoryDescriptor(tensor);
        if (bytes_override > req.source.size_bytes)
        {
            return TransferResult::fail(
                method,
                "activation byte override exceeds tensor allocation");
        }
        if (bytes_override != 0)
            req.source.size_bytes = bytes_override;
        req.source.device = src;
        if (src.is_gpu())
        {
            req.source.device_ptr =
                tensor->getOrAllocateDeviceBuffer(src);
            if (!req.source.device_ptr)
            {
                return TransferResult::fail(
                    method,
                    "authoritative source has no activation buffer on " +
                        src.toString());
            }
        }
        req.target_device = target_device;
        req.method = method;

        // For activation transfer, ensure destination buffer exists
        if (target_device.is_gpu())
        {
            void *dst_ptr = tensor->getOrAllocateDeviceBuffer(target_device);
            if (!dst_ptr)
                return TransferResult::fail(method,
                                            "failed to allocate device buffer for activation transfer to " +
                                                target_device.toString());
            req.target_ptr = dst_ptr;
        }

        auto result = execute(req);

        if (result.success)
        {
            if (target_device.is_gpu())
            {
                void *target_ptr = req.target_ptr;
                if (!tensor->gpu_device_.has_value() ||
                    *tensor->gpu_device_ != target_device)
                {
                    if (tensor->gpu_device_.has_value() &&
                        tensor->gpu_data_ptr_)
                    {
                        tensor->secondary_device_buffers_
                            [TensorBase::packDeviceId(*tensor->gpu_device_)] =
                            tensor->gpu_data_ptr_;
                    }
                    tensor->gpu_device_ = target_device;
                    tensor->gpu_data_ptr_ = target_ptr;
                    tensor->secondary_device_buffers_.erase(
                        TensorBase::packDeviceId(target_device));
                }
                publishDeviceWrite(
                    tensor,
                    target_device,
                    requireTransferStream(
                        target_device,
                        "TransferEngine::transferActivation publication"));
            }
            else
            {
                tensor->applyCoherenceOp_(CoherenceOp::DOWNLOAD);
            }
        }

        return result;
    }

    TransferResult TransferEngine::copyActivation(TensorBase *src, TensorBase *dst,
                                                  DeviceId dst_device, size_t bytes)
    {
        if (!src || !dst)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor in copyActivation");
        src = src->transferStorageOwner();
        dst = dst->transferStorageOwner();
        if (!src || !dst ||
            src->transferStorageOwner() != src ||
            dst->transferStorageOwner() != dst)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "activation has no canonical transfer-storage owner");
        }
        if (bytes == 0)
            return TransferResult::ok(TransferMethod::NOOP);

        // ----------------------------------------------------------------------
        // Host or mapped destination: a plain host-side copy is correct and
        // cheapest. Mapped memory shares host/device storage, so writing the
        // host side is immediately visible to the device — no transfer needed.
        // ----------------------------------------------------------------------
        if (dst_device.is_cpu() || dst->is_mapped_ || src->is_mapped_)
        {
            const void *host_src = src->data();   // ensures src is host-valid
            void *host_dst = dst->mutable_data(); // host (or mapped) storage
            if (!host_src || !host_dst)
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                            "null host pointer in copyActivation host path");
            std::memcpy(host_dst, host_src, bytes);
            if (dst_device.is_gpu())
            {
                /*
                 * Mapped host memory is immediately visible to the GPU, but the
                 * coherence publication still needs a concrete completion
                 * token. Record it on the destination backend's owned default
                 * stream so later GPU consumers never inherit eventless device
                 * authority from this host-visible transfer boundary.
                 */
                publishCompletedDeviceWrite(dst, dst_device);
            }
            return TransferResult::ok(dst->is_mapped_ ? TransferMethod::MAPPED_NOOP
                                                      : TransferMethod::DEVICE_TO_HOST);
        }

        // ----------------------------------------------------------------------
        // GPU destination: ensure a device buffer exists. We deliberately do NOT
        // upload host data here (the buffer is about to be overwritten by the
        // copy below).
        // ----------------------------------------------------------------------
        void *dst_ptr = dst->getOrAllocateDeviceBuffer(dst_device);
        if (!dst_ptr)
            return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                        "failed to allocate dst device buffer on " + dst_device.toString());

        // Locate the authoritative source data. NOTE: we use getAuthoritativeDevice()
        // rather than gpu_device_: after a cross-vendor transferActivation(), the
        // tensor's primary gpu_device_/gpu_data_ptr_ still point at the producing
        // GPU (e.g. CUDA), while the authoritative copy lives in a secondary buffer
        // on the consumer GPU (e.g. ROCm).
        const auto auth = src->getAuthoritativeDevice();
        const bool src_on_gpu = auth.has_value() && auth->is_gpu();
        const DeviceId src_device = src_on_gpu ? *auth : DeviceId::cpu();

        TransferResult result;

        // Decide whether a direct device-to-device path exists for this pair.
        // Same physical GPU is always direct (intra-VRAM memcpy). Different GPUs
        // of the SAME vendor go through the collective backend (NCCL/RCCL/peer
        // DMA). Only cross-vendor pairs (CUDA↔ROCm) have no direct path and must
        // bounce through the host.
        const bool same_physical_gpu = src_on_gpu && src_device == dst_device;
        const bool same_vendor_diff_gpu =
            src_on_gpu && !same_physical_gpu && src_device.type == dst_device.type;

        if (same_physical_gpu)
        {
            // Same physical GPU: pure device-to-device copy — no host bounce.
            void *src_ptr = src->getOrAllocateDeviceBuffer(src_device);
            IBackend *backend = resolveBackend(dst_device);
            if (!backend)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "no backend for " + dst_device.toString());
            if (!src_ptr)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "source has no device buffer on " + dst_device.toString());
            void *const destination_stream =
                requireTransferStream(
                    dst_device,
                    "TransferEngine::copyActivation same-device copy");
            if (!backend->deviceToDevice(
                    dst_ptr,
                    src_ptr,
                    bytes,
                    dst_device.gpu_ordinal(),
                    destination_stream))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "deviceToDevice failed on " + dst_device.toString());
            result = TransferResult::ok(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
        }
        else if (same_vendor_diff_gpu)
        {
            // Same-vendor, different GPU (e.g. cuda:0 → cuda:1): use the
            // collective backend's peer copy (NCCL/RCCL or peer DMA). No host
            // bounce — the data moves directly across the PCIe/NVLink fabric.
            void *src_ptr = src->getOrAllocateDeviceBuffer(src_device);
            auto *router = GlobalBackendRouter::get();
            ICollectiveBackend *backend =
                router ? router->getBackendForCopy(src_device, dst_device) : nullptr;
            if (!src_ptr)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "source has no device buffer on " + src_device.toString());
            if (!backend || !backend->supportsCopy(src_device, dst_device))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "no collective backend supports peer copy " +
                                                src_device.toString() + " -> " + dst_device.toString());
            if (!backend->copy(dst_ptr, dst_device, src_ptr, src_device, bytes))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "peer copy failed " + src_device.toString() +
                                                " -> " + dst_device.toString());
            result = TransferResult::ok(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
        }
        else
        {
            // Cross-vendor GPU pair (CUDA↔ROCm) or host-resident source: there
            // is no direct device path across vendors, so stage through the
            // source tensor's own host buffer and upload into the dst device
            // buffer. This is what makes heterogeneous CUDA↔ROCm PP work.
            const void *host_src = nullptr;
            if (src_on_gpu)
            {
                void *src_host = src->raw_host_data_ptr();
                void *src_dev = src->getOrAllocateDeviceBuffer(src_device);
                IBackend *src_backend = resolveBackend(src_device);
                if (!src_host || !src_dev || !src_backend)
                    return TransferResult::fail(TransferMethod::HOST_STAGED,
                                                "missing src host/device buffer or backend for staged copy");
                void *const source_stream =
                    requireTransferStream(
                        src_device,
                        "TransferEngine::copyActivation staged D2H");
                if (!src_backend->deviceToHost(
                        src_host,
                        src_dev,
                        bytes,
                        src_device.gpu_ordinal(),
                        source_stream))
                    return TransferResult::fail(TransferMethod::HOST_STAGED,
                                                "D2H step failed in copyActivation");
                host_src = src_host;
            }
            else
            {
                host_src = src->data(); // host-resident source
            }

            IBackend *dst_backend = resolveBackend(dst_device);
            if (!dst_backend || !host_src)
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "missing dst backend or host src in copyActivation");
            void *const destination_stream =
                requireTransferStream(
                    dst_device,
                    "TransferEngine::copyActivation staged H2D");
            if (!dst_backend->hostToDevice(
                    dst_ptr,
                    host_src,
                    bytes,
                    dst_device.gpu_ordinal(),
                    destination_stream))
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "H2D step failed in copyActivation");
            result = TransferResult::ok(src_on_gpu ? TransferMethod::HOST_STAGED
                                                   : TransferMethod::HOST_TO_DEVICE);
        }

        // ----------------------------------------------------------------------
        // Promote dst_ptr to the primary device buffer if it currently lives in
        // the secondary map (so later gpu_data_ptr() returns the buffer we just
        // wrote), then mark the destination authoritative on dst_device.
        // ----------------------------------------------------------------------
        if (!dst->gpu_device_.has_value() || *dst->gpu_device_ != dst_device)
        {
            // Preserve any existing primary buffer as a secondary so it is not leaked.
            if (dst->gpu_device_.has_value() && dst->gpu_data_ptr_)
                dst->secondary_device_buffers_[TensorBase::packDeviceId(*dst->gpu_device_)] =
                    dst->gpu_data_ptr_;
            dst->gpu_device_ = dst_device;
            dst->gpu_data_ptr_ = dst_ptr;
            dst->secondary_device_buffers_.erase(TensorBase::packDeviceId(dst_device));
        }
        /*
         * copyActivation() currently completes its selected transport before
         * returning. Publish a fresh backend event after that boundary rather
         * than erasing completion metadata with a plain coherence transition.
         * The event gives every later device consumer one uniform dependency
         * contract regardless of whether the bytes arrived via intra-device,
         * peer, or deliberately heterogeneous host-staged transport.
         */
        publishDeviceWrite(
            dst,
            dst_device,
            requireTransferStream(
                dst_device,
                "TransferEngine::copyActivation publication"));

        return result;
    }

    // ============================================================================
    // Private: execute*() methods
    // ============================================================================

    TransferResult TransferEngine::executeHostToDevice(const TransferRequest &req)
    {
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "source host_ptr is null");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null");

        IBackend *backend = resolveBackend(req.target_device);
        if (!backend)
            return TransferResult::fail(req.method,
                                        "no backend for device " + req.target_device.toString());

        bool ok = backend->hostToDevice(
            req.target_ptr,
            req.source.host_ptr,
            req.source.size_bytes,
            req.target_device.ordinal,
            requireTransferStream(
                req.target_device,
                "TransferEngine::executeHostToDevice"));
        if (!ok)
            return TransferResult::fail(req.method, "hostToDevice failed");

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeDeviceToHost(const TransferRequest &req)
    {
        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null");
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "destination host_ptr is null (no host buffer)");

        IBackend *backend = resolveBackend(req.source.device);
        if (!backend)
            return TransferResult::fail(req.method,
                                        "no backend for source device " + req.source.device.toString());

        bool ok = backend->deviceToHost(
            req.source.host_ptr,
            req.source.device_ptr,
            req.source.size_bytes,
            req.source.device.ordinal,
            requireTransferStream(
                req.source.device,
                "TransferEngine::executeDeviceToHost"));
        if (!ok)
            return TransferResult::fail(req.method, "deviceToHost failed");

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeDeviceToDeviceSameBackend(const TransferRequest &req)
    {
        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null");

        /*
         * Same-vendor movement is a collective-backend responsibility. NCCL
         * and RCCL select their best available P2P transport; silently bouncing
         * through host memory here would hide a broken device path and violate
         * the device-owned pipeline contract.
         */
        auto *router = GlobalBackendRouter::get();
        ICollectiveBackend *backend =
            router
                ? router->getBackendForCopy(
                      req.source.device,
                      req.target_device)
                : nullptr;
        if (!backend ||
            !backend->supportsCopy(
                req.source.device,
                req.target_device))
        {
            return TransferResult::fail(
                req.method,
                "no direct collective backend supports " +
                    req.source.device.toString() + " -> " +
                    req.target_device.toString());
        }
        if (!backend->copy(
                req.target_ptr,
                req.target_device,
                req.source.device_ptr,
                req.source.device,
                req.source.size_bytes))
        {
            return TransferResult::fail(
                req.method,
                "collective backend copy failed " +
                    req.source.device.toString() + " -> " +
                    req.target_device.toString());
        }

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeHostStaged(const TransferRequest &req)
    {
        // HOST_STAGED: D2H from source GPU → memcpy → H2D to target GPU
        // Used for cross-vendor transfers.

        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null for host staged");
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "host buffer needed for host staged bounce");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null for host staged");

        IBackend *src_backend = resolveBackend(req.source.device);
        if (!src_backend)
            return TransferResult::fail(req.method,
                                        "no backend for source " + req.source.device.toString());

        // Step 1: D2H
        bool d2h = src_backend->deviceToHost(
            req.source.host_ptr, req.source.device_ptr,
            req.source.size_bytes, req.source.device.ordinal,
            requireTransferStream(
                req.source.device,
                "TransferEngine::executeHostStaged source"));
        if (!d2h)
            return TransferResult::fail(req.method, "D2H step of host staged failed");

        // Step 2: H2D to target
        IBackend *dst_backend = resolveBackend(req.target_device);
        if (!dst_backend)
            return TransferResult::fail(req.method,
                                        "no backend for target " + req.target_device.toString());

        bool h2d = dst_backend->hostToDevice(
            req.target_ptr, req.source.host_ptr,
            req.source.size_bytes, req.target_device.ordinal,
            requireTransferStream(
                req.target_device,
                "TransferEngine::executeHostStaged destination"));
        if (!h2d)
            return TransferResult::fail(req.method, "H2D step of host staged failed");

        return TransferResult::ok(req.method);
    }

    // ============================================================================
    // Backend resolution
    // ============================================================================

    bool TransferEngine::waitForEventWithProxy(IBackend *backend, void *event, int device_id,
                                               const DeviceId &gpu_device)
    {
        return backend->waitForEvent(event, device_id);
    }

    void TransferEngine::waitForPendingHostSourceUseLocked(TensorBase *tensor)
    {
        if (!tensor ||
            tensor->device_completion_purpose_ !=
                TensorBase::CompletionEventPurpose::HOST_TO_DEVICE_SOURCE_USE)
        {
            return;
        }
        if (!tensor->device_completion_event_ ||
            !tensor->event_device_.has_value() ||
            !tensor->event_device_->is_gpu())
        {
            throw std::runtime_error(
                "Queued H2D host-source use has no exact completion event");
        }

        IBackend *const backend =
            tensor->resolveBackend(*tensor->event_device_);
        if (!backend ||
            backend->backendDeviceType() != tensor->event_device_->type)
        {
            throw std::runtime_error(
                "Queued H2D host-source event has no matching backend on " +
                tensor->event_device_->toString());
        }
        if (!waitForEventWithProxy(
                backend,
                tensor->device_completion_event_,
                tensor->event_device_->gpu_ordinal(),
                *tensor->event_device_))
        {
            throw std::runtime_error(
                "Queued H2D host-source completion event failed on " +
                tensor->event_device_->toString());
        }

        /*
         * The host wait proves both DMA completion and device visibility. The
         * event no longer carries an outstanding lifetime, so retire it rather
         * than letting a later generation accidentally reuse its identity.
         */
        tensor->retireCompletionEvent_();
    }

    // ============================================================================
    // uploadFull — full ensureOnDevice lifecycle (called with coherence_mutex_ held)
    // ============================================================================

    TransferResult TransferEngine::uploadFull(TensorBase *tensor, DeviceId target_device, void *stream)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        // ===== HOST-RESIDENT FAST PATH =====
        // Tensors marked HOST_RESIDENT are consumed on host only (e.g., embedding
        // tables that get repacked into a device workspace by the kernel).
        // Skip device allocation and upload entirely.
        if (tensor->memoryResidency() == MemoryResidency::HOST_RESIDENT)
            return TransferResult::ok(TransferMethod::NOOP);

        const bool trace = debugEnv().rocm.trace_coherence;
        auto overall_start = std::chrono::high_resolution_clock::now();

        // ===== ZERO-COPY MAPPED MEMORY FAST PATH =====
        if (tensor->is_mapped_ && tensor->mapped_device_ptr_ != nullptr)
        {
            if (!tensor->gpu_device_.has_value() || *tensor->gpu_device_ != target_device)
            {
                tensor->gpu_device_ = target_device;
            }
            if (tensor->gpu_data_ptr_ != tensor->mapped_device_ptr_)
            {
                tensor->gpu_data_ptr_ = tensor->mapped_device_ptr_;
            }
            tensor->setCoherenceState_(TensorCoherenceState::MAPPED);

            if (trace)
            {
                LOG_TRACE("[TransferEngine::uploadFull] ZERO-COPY: Tensor is mapped, no memcpy needed");
            }
            return TransferResult::ok(TransferMethod::MAPPED_NOOP);
        }

        // ===== ALREADY ON TARGET DEVICE (with event wait) =====
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value() &&
            *tensor->gpu_device_ == target_device && ::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            if (tensor->device_completion_event_)
            {
                IBackend *backend = tensor->resolveBackend(target_device);
                if (backend)
                {
                    const int backend_device_id = target_device.gpu_ordinal();
                    if (stream)
                    {
                        // Non-blocking: make the consuming stream wait for the event.
                        // Same-stream waits are no-ops in hardware (ordering is implicit).
                        // Cross-stream waits correctly serialize without blocking the CPU.
                        if (!backend->streamWaitEvent(stream, tensor->device_completion_event_, backend_device_id))
                        {
                            return TransferResult::fail(
                                TransferMethod::NOOP,
                                "Stream event wait failed for tensor '" +
                                    (tensor->debug_name_.empty()
                                         ? std::string("(unnamed)")
                                         : tensor->debug_name_) +
                                    "' on " + target_device.toString() +
                                    "; refusing a host-blocking fallback");
                        }
                    }
                }
            }
            return TransferResult::ok(TransferMethod::NOOP);
        }

        // Get backend for target device
        IBackend *target_backend = tensor->resolveBackend(target_device);
        if (!target_backend)
        {
            return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                        "No backend available for device " + target_device.toString());
        }

        int backend_device_id = target_device.gpu_ordinal();

        // ===== DEVICE MIGRATION (secondary buffers) =====
        LOG_TRACE("[TransferEngine::uploadFull] tensor=" << static_cast<void *>(tensor)
                                                         << " target_device=" << target_device.toString()
                                                         << " gpu_data_ptr_=" << tensor->gpu_data_ptr_
                                                         << " gpu_device_=" << (tensor->gpu_device_.has_value() ? tensor->gpu_device_->toString() : "none")
                                                         << " device_completion_event_=" << tensor->device_completion_event_);
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value() && *tensor->gpu_device_ != target_device)
        {
            /*
             * A device migration may retire the old event and eventually reuse
             * the host allocation as a source for the new device. Quiesce only
             * an outstanding H2D source use before changing event ownership.
             */
            waitForPendingHostSourceUseLocked(tensor);
            DeviceId old_device = *tensor->gpu_device_;
            LOG_TRACE("[TransferEngine::uploadFull] Device migration: " << tensor->gpu_device_->toString()
                                                                        << " -> " << target_device.toString());

            const int target_key = TensorBase::packDeviceId(target_device);
            auto sec_it = tensor->secondary_device_buffers_.find(target_key);
            if (sec_it != tensor->secondary_device_buffers_.end() && sec_it->second != nullptr)
            {
                // Preserve current primary in secondary map
                int old_key = TensorBase::packDeviceId(*tensor->gpu_device_);
                if (tensor->secondary_device_buffers_.find(old_key) == tensor->secondary_device_buffers_.end())
                {
                    tensor->secondary_device_buffers_[old_key] = tensor->gpu_data_ptr_;
                }

                // Promote target secondary buffer to primary
                void *promoted_ptr = sec_it->second;
                tensor->secondary_device_buffers_.erase(sec_it);

                tensor->gpu_data_ptr_ = promoted_ptr;
                tensor->gpu_device_ = target_device;
                tensor->setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);

                if (tensor->device_completion_event_)
                {
                    IBackend *old_backend = tensor->resolveBackend(old_device);
                    int old_backend_device_id = old_device.gpu_ordinal();
                    if (old_backend)
                    {
                        old_backend->destroyEvent(tensor->device_completion_event_, old_backend_device_id);
                    }
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                    tensor->device_completion_purpose_ =
                        TensorBase::CompletionEventPurpose::NONE;
                }

                LOG_TRACE("[TransferEngine::uploadFull] Promoted secondary buffer to primary for "
                          << target_device.toString() << " ptr=" << promoted_ptr);
            }
            else
            {
                // No existing target secondary buffer. Park current primary.
                int old_key = TensorBase::packDeviceId(*tensor->gpu_device_);
                if (tensor->secondary_device_buffers_.find(old_key) == tensor->secondary_device_buffers_.end())
                {
                    tensor->secondary_device_buffers_[old_key] = tensor->gpu_data_ptr_;
                }

                if (tensor->device_completion_event_)
                {
                    IBackend *old_backend = tensor->resolveBackend(*tensor->gpu_device_);
                    int old_backend_device_id = tensor->gpu_device_->gpu_ordinal();
                    if (old_backend)
                    {
                        LOG_TRACE("[TransferEngine::uploadFull] Destroying old completion event on device "
                                  << tensor->gpu_device_->toString() << " before migrating to " << target_device.toString());
                        old_backend->destroyEvent(tensor->device_completion_event_, old_backend_device_id);
                    }
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                    tensor->device_completion_purpose_ =
                        TensorBase::CompletionEventPurpose::NONE;
                }

                tensor->gpu_data_ptr_ = nullptr;
                tensor->gpu_device_.reset();
                tensor->applyCoherenceOp_(CoherenceOp::RELEASE_DEVICE);

                LOG_TRACE("[TransferEngine::uploadFull] Parked previous primary buffer for "
                          << old_device.toString() << " and allocating fresh buffer for "
                          << target_device.toString());
            }
        }

        // ===== ALLOCATE ON TARGET DEVICE =====
        size_t bytes = tensor->byte_size();
        if (!tensor->gpu_data_ptr_)
        {
            auto alloc_start = std::chrono::high_resolution_clock::now();
            tensor->gpu_data_ptr_ = target_backend->allocate(bytes, backend_device_id);
            auto alloc_end = std::chrono::high_resolution_clock::now();
            auto alloc_us = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();

            if (trace)
            {
                LOG_TRACE("[TransferEngine::uploadFull] backend->allocate(" << bytes << " bytes) took " << alloc_us << " us");
            }

            LOG_TRACE("[GPU_ALLOC] tensor=" << static_cast<void *>(tensor)
                                            << " name=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                                            << " gpu_ptr=" << tensor->gpu_data_ptr_ << " bytes=" << bytes
                                            << " device=" << target_device.toString()
                                            << " ordinal=" << backend_device_id);

            if (!tensor->gpu_data_ptr_)
            {
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "Failed to allocate " + std::to_string(bytes) + " bytes on " + target_device.toString());
            }
            tensor->gpu_device_ = target_device;
            tensor->setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);

            // Pin host memory for fast DMA transfers
            auto pin_start = std::chrono::high_resolution_clock::now();
            tensor->ensureHostPinned();
            auto pin_end = std::chrono::high_resolution_clock::now();
            auto pin_us = std::chrono::duration_cast<std::chrono::microseconds>(pin_end - pin_start).count();

            if (trace && pin_us > 100)
            {
                LOG_TRACE("[TransferEngine::uploadFull] ensureHostPinned() took " << pin_us << " us");
            }
        }

        // ===== H2D UPLOAD =====
        if (!::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            if (!::llaminar2::isHostValid(tensor->coherence_state_))
            {
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "COHERENCE ERROR: Both host and device are invalid");
            }

            const void *src = tensor->raw_host_data_ptr();
            if (!src)
            {
                target_backend->free(tensor->gpu_data_ptr_, backend_device_id);
                tensor->gpu_data_ptr_ = nullptr;
                tensor->gpu_device_.reset();
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE, "Host data pointer is null");
            }

            // Transfer tracing
            const auto &trace_cfg = debugEnv().transfer_tracing;
            if (trace_cfg.enabled && !trace_cfg.only_d2h && bytes >= trace_cfg.min_bytes)
            {
                std::ostringstream msg;
                msg << "[TRANSFER TRACE] H2D transfer: " << bytes << " bytes"
                    << ", tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                    << ", shape=[" << tensor->rows() << "x" << tensor->cols() << "]"
                    << ", device=" << target_device.toString();

                if (trace_cfg.include_stacktrace)
                {
                    msg << "\n"
                        << captureStackTrace(2, 16);
                }

                if (trace_cfg.throw_on_transfer)
                {
                    throw TransferViolationException(msg.str(), captureStackTrace(2, 32));
                }
                else
                {
                    LOG_WARN(msg.str());
                }
            }

            void *const upload_stream =
                stream
                    ? stream
                    : requireTransferStream(
                          target_device,
                          "TransferEngine::uploadFull");
            /*
             * Order an overwrite after any previous producer of this device
             * allocation. This is a device-side edge; the submitting CPU never
             * waits for the prior generation.
             */
            if (tensor->device_completion_event_)
            {
                if (!tensor->event_device_.has_value() ||
                    *tensor->event_device_ != target_device)
                {
                    return TransferResult::fail(
                        TransferMethod::HOST_TO_DEVICE,
                        "H2D upload found completion-event ownership on the wrong device");
                }
                if (!target_backend->streamWaitEvent(
                        upload_stream,
                        tensor->device_completion_event_,
                        backend_device_id))
                {
                    return TransferResult::fail(
                        TransferMethod::HOST_TO_DEVICE,
                        "H2D upload could not join the prior device producer event");
                }
            }

            bool created_event = false;
            if (!tensor->device_completion_event_)
            {
                tensor->device_completion_event_ =
                    target_backend->createEvent(backend_device_id);
                if (!tensor->device_completion_event_)
                {
                    return TransferResult::fail(
                        TransferMethod::HOST_TO_DEVICE,
                        "H2D upload could not create its completion event");
                }
                tensor->event_device_ = target_device;
                created_event = true;
            }

            auto h2d_start = std::chrono::high_resolution_clock::now();
            bool h2d_ok = target_backend->hostToDeviceOnStream(
                tensor->gpu_data_ptr_,
                src,
                bytes,
                backend_device_id,
                upload_stream);
            auto h2d_end = std::chrono::high_resolution_clock::now();
            auto h2d_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(h2d_end - h2d_start).count();
            auto h2d_us = h2d_ns / 1000;

            if (trace_cfg.enabled)
            {
                trace_cfg.recordH2D(bytes);
            }

            if (trace)
            {
                LOG_TRACE("[TransferEngine::uploadFull] queued hostToDevice(" << bytes
                                                                               << " bytes) in "
                                                                               << h2d_us << " us");
            }

            if (!h2d_ok)
            {
                if (created_event)
                {
                    target_backend->destroyEvent(
                        tensor->device_completion_event_,
                        backend_device_id);
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                }
                return TransferResult::fail(
                    TransferMethod::HOST_TO_DEVICE,
                    "asynchronous hostToDevice enqueue failed");
            }

            if (!target_backend->recordEvent(
                    tensor->device_completion_event_,
                    backend_device_id,
                    upload_stream))
            {
                /*
                 * The runtime accepted DMA but failed to publish the only safe
                 * source-lifetime boundary. Continuing could free or overwrite
                 * pinned host bytes still in use, so this is unrecoverable.
                 */
                LOG_ERROR("[TransferEngine::uploadFull] Accepted H2D copy could not publish its exact completion event on "
                          << target_device.toString());
                std::terminate();
            }

            tensor->last_joined_completion_event_ = nullptr;
            tensor->last_joined_consumer_stream_ = nullptr;
            tensor->device_completion_purpose_ =
                TensorBase::CompletionEventPurpose::HOST_TO_DEVICE_SOURCE_USE;
            tensor->applyCoherenceOp_(CoherenceOp::UPLOAD);
            tensor->authoritative_device_.reset();
            TransferProfiler::recordH2D(bytes);

            // GPU_ONLY policy: free host data now that device has it
            if (tensor->memoryResidency() == MemoryResidency::GPU_ONLY &&
                !tensor->is_raw_data_released())
            {
                waitForPendingHostSourceUseLocked(tensor);
                tensor->release_host_weight_data();
                tensor->setCoherenceState_(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
                tensor->authoritative_device_ = target_device;
            }

            LOG_TRACE("[TransferEngine::uploadFull] Uploaded " << bytes
                                                               << " bytes to device " << target_device.toString()
                                                               << " (backend device ID: " << backend_device_id << ")");
        }

        auto overall_end = std::chrono::high_resolution_clock::now();
        auto overall_us = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start).count();
        if (trace && overall_us > 1000)
        {
            LOG_TRACE("[TransferEngine::uploadFull] TOTAL took " << overall_us << " us for " << bytes << " bytes");
        }

        return TransferResult::ok(TransferMethod::HOST_TO_DEVICE);
    }

    // ============================================================================
    // downloadFull — full ensureOnHost lifecycle (called with coherence_mutex_ held)
    // ============================================================================

    TransferResult TransferEngine::downloadFull(TensorBase *tensor, void *stream)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        // ===== ZERO-COPY MAPPED MEMORY PATH =====
        if (tensor->is_mapped_)
        {
            if (tensor->mapped_needs_sync_ && tensor->gpu_device_.has_value())
            {
                IBackend *backend = tensor->resolveBackend(*tensor->gpu_device_);
                if (backend)
                {
                    int backend_device_id = tensor->gpu_device_->gpu_ordinal();

                    auto t0 = std::chrono::high_resolution_clock::now();

                    if (tensor->device_completion_event_)
                    {
                        LOG_TRACE("[TransferEngine::downloadFull] ZERO-COPY: Waiting on completion event");
                        if (!waitForEventWithProxy(backend, tensor->device_completion_event_, backend_device_id, *tensor->gpu_device_))
                        {
                            return TransferResult::fail(
                                TransferMethod::MAPPED_NOOP,
                                "Mapped tensor completion event wait failed");
                        }
                    }
                    else
                    {
                        return TransferResult::fail(
                            TransferMethod::MAPPED_NOOP,
                            "Mapped GPU tensor has no completion event");
                    }

                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (elapsed_ms > 1.0)
                    {
                        LOG_WARN("[TransferEngine::downloadFull] MAPPED SYNC took " << elapsed_ms << " ms"
                                                                                    << " (event=" << (tensor->device_completion_event_ ? "yes" : "NO") << ")");
                    }
                }
                tensor->mapped_needs_sync_ = false;
            }

            tensor->setCoherenceState_(TensorCoherenceState::MAPPED);
            tensor->authoritative_device_ = std::nullopt;
            return TransferResult::ok(TransferMethod::MAPPED_NOOP);
        }

        // ===== HOST ALREADY VALID =====
        if (::llaminar2::isHostValid(tensor->coherence_state_))
        {
            LOG_TRACE("[TransferEngine::downloadFull] Host already valid, skipping sync");
            return TransferResult::ok(TransferMethod::NOOP);
        }

        // Device must be valid if host is invalid
        if (!::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                        "COHERENCE ERROR: Both host and device are invalid");
        }

        // ===== STANDARD GPU D2H =====
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value())
        {
            IBackend *backend = tensor->resolveBackend(*tensor->gpu_device_);
            if (!backend)
            {
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                            "No backend available for device " + tensor->gpu_device_->toString());
            }

            int backend_device_id = tensor->gpu_device_->gpu_ordinal();

            size_t bytes = tensor->byte_size();
            void *dst = tensor->raw_host_data_ptr();
            if (!dst)
            {
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST, "Host data pointer is null");
            }

            if (!stream && !tensor->device_completion_event_)
            {
                return TransferResult::fail(
                    TransferMethod::DEVICE_TO_HOST,
                    "GPU tensor has no completion event or explicit producer stream");
            }

            // Transfer tracing for D2H debugging
            const auto &trace_cfg = debugEnv().transfer_tracing;
            if (trace_cfg.enabled && bytes >= trace_cfg.min_bytes)
            {
                std::ostringstream msg;
                msg << "[TRANSFER TRACE] D2H transfer: " << bytes << " bytes"
                    << ", tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                    << ", shape=[" << tensor->rows() << "x" << tensor->cols() << "]"
                    << ", device=" << tensor->gpu_device_->toString();

                if (trace_cfg.include_stacktrace)
                {
                    msg << "\n"
                        << captureStackTrace(2, 16);
                }

                if (trace_cfg.throw_on_transfer)
                {
                    throw TransferViolationException(msg.str(), captureStackTrace(2, 32));
                }
                else
                {
                    LOG_WARN(msg.str());
                }
            }

            LOG_TRACE("[TransferEngine::downloadFull] ATTEMPTING D2H: "
                      << "tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                      << " gpu_data_ptr=" << static_cast<const void *>(tensor->gpu_data_ptr_)
                      << " dst=" << static_cast<const void *>(dst)
                      << " bytes=" << bytes
                      << " device=" << tensor->gpu_device_->toString()
                      << " backend_device_id=" << backend_device_id);

            void *const download_stream =
                stream
                    ? stream
                    : requireTransferStream(
                          *tensor->gpu_device_,
                          "TransferEngine::downloadFull");

            /*
             * A caller-provided stream is the exact producer stream and already
             * carries the source dependency. Otherwise import the published
             * producer event onto the dedicated transfer stream. Neither path
             * blocks the host before the D2H copy is submitted.
             */
            if (!stream)
            {
                if (!tensor->event_device_.has_value() ||
                    *tensor->event_device_ != *tensor->gpu_device_ ||
                    !backend->streamWaitEvent(
                        download_stream,
                        tensor->device_completion_event_,
                        backend_device_id))
                {
                    return TransferResult::fail(
                        TransferMethod::DEVICE_TO_HOST,
                        "D2H transfer could not join the exact device producer event");
                }
            }

            bool created_event = false;
            if (!tensor->device_completion_event_)
            {
                tensor->device_completion_event_ =
                    backend->createEvent(backend_device_id);
                if (!tensor->device_completion_event_)
                {
                    return TransferResult::fail(
                        TransferMethod::DEVICE_TO_HOST,
                        "D2H transfer could not create its host-publication event");
                }
                tensor->event_device_ = *tensor->gpu_device_;
                created_event = true;
            }

            auto d2h_start = std::chrono::high_resolution_clock::now();
            bool d2h_ok = backend->deviceToHostOnStream(
                dst,
                tensor->gpu_data_ptr_,
                bytes,
                backend_device_id,
                download_stream);
            if (!d2h_ok)
            {
                if (created_event)
                {
                    backend->destroyEvent(
                        tensor->device_completion_event_,
                        backend_device_id);
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                }
                return TransferResult::fail(
                    TransferMethod::DEVICE_TO_HOST,
                    "asynchronous deviceToHost enqueue failed");
            }
            if (!backend->recordEvent(
                    tensor->device_completion_event_,
                    backend_device_id,
                    download_stream))
            {
                LOG_ERROR("[TransferEngine::downloadFull] Accepted D2H copy could not publish its exact host-completion event on "
                          << tensor->gpu_device_->toString());
                std::terminate();
            }
            tensor->last_joined_completion_event_ = nullptr;
            tensor->last_joined_consumer_stream_ = nullptr;
            if (!waitForEventWithProxy(
                    backend,
                    tensor->device_completion_event_,
                    backend_device_id,
                    *tensor->gpu_device_))
            {
                return TransferResult::fail(
                    TransferMethod::DEVICE_TO_HOST,
                    "D2H host-publication event wait failed; completion event "
                    "is invalid or could not be observed");
            }
            auto d2h_end = std::chrono::high_resolution_clock::now();
            auto d2h_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(d2h_end - d2h_start).count();

            // Optional transfer trace diagnostic. This only samples the first
            // few floats, so keep it out of normal logs and validation output.
            if (trace_cfg.enabled)
            {
                const float *fp = static_cast<const float *>(dst);
                size_t check_count = std::min(bytes / sizeof(float), static_cast<size_t>(8));
                bool all_zero = true;
                for (size_t i = 0; i < check_count; ++i)
                {
                    if (fp[i] != 0.0f)
                    {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero && check_count > 0 && bytes >= 1024)
                {
                    LOG_TRACE("[TransferEngine::downloadFull] D2H leading sample is zero: tensor="
                              << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                              << " bytes=" << bytes
                              << " device=" << tensor->gpu_device_->toString()
                              << " gpu_ptr=" << static_cast<const void *>(tensor->gpu_data_ptr_)
                              << " dst=" << static_cast<const void *>(dst)
                              << " d2h_ns=" << d2h_ns
                              << " had_event=" << (tensor->device_completion_event_ ? "yes" : "no"));
                }
            }

            TransferProfiler::recordD2H(bytes, static_cast<uint64_t>(d2h_ns));

            if (trace_cfg.enabled)
            {
                trace_cfg.recordD2H(bytes);
            }

            tensor->applyCoherenceOp_(CoherenceOp::DOWNLOAD);
            tensor->authoritative_device_ = std::nullopt;
            tensor->retireCompletionEvent_();

            LOG_TRACE("[TransferEngine::downloadFull] Downloaded " << bytes
                                                                   << " bytes from device " << tensor->gpu_device_->toString()
                                                                   << " (backend device ID: " << backend_device_id << ")");
        }

        return TransferResult::ok(TransferMethod::DEVICE_TO_HOST);
    }

    IBackend *TransferEngine::resolveBackend(DeviceId device) const
    {
        if (resolve_)
            return resolve_(device);
        return getBackendFor(device);
    }

    // ============================================================================
    // Transfer tracing
    // ============================================================================

    void TransferEngine::traceTransfer(const TransferRequest &req, const TransferResult &result) const
    {
        const auto &tracing = debugEnv().transfer_tracing;
        if (!tracing.enabled)
            return;

        auto method_str = to_string(req.method);
        auto src_str = req.source.device.toString();
        auto dst_str = req.target_device.toString();

        if (result.success)
        {
            LOG_TRACE("[TransferEngine] " << method_str
                                          << " " << src_str << " → " << dst_str
                                          << " (" << req.source.size_bytes << " bytes"
                                          << ", " << result.elapsed_ns / 1000 << " μs)");
        }
        else
        {
            LOG_WARN("[TransferEngine] FAILED " << method_str
                                                << " " << src_str << " → " << dst_str
                                                << ": " << result.error);
        }
    }

    // ============================================================================
    // MemoryDescriptor factory
    // ============================================================================

    MemoryDescriptor makeMemoryDescriptor(const TensorBase *tensor)
    {
        MemoryDescriptor desc;

        if (!tensor)
            return desc;
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
            return desc;

        // Host pointer
        desc.host_ptr = const_cast<void *>(tensor->raw_data());
        desc.size_bytes = tensor->byte_size();

        // GPU pointer and device
        if (tensor->gpu_device_.has_value())
        {
            desc.device = tensor->gpu_device_.value();
            desc.device_ptr = tensor->gpu_data_ptr_;
        }
        else
        {
            desc.device = DeviceId::cpu();
        }

        // Memory residency from canonical source
        desc.residency = tensor->memoryResidency();
        if (desc.residency == MemoryResidency::MAPPED)
        {
            desc.mapped_host_ptr = tensor->mapped_host_ptr_;
            desc.mapped_device_ptr = tensor->mapped_device_ptr_;
        }

        return desc;
    }

} // namespace llaminar2
