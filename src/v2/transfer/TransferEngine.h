/**
 * @file TransferEngine.h
 * @brief Public tensor movement, coherence, and event-publication authority.
 *
 * Production tensor callers use TransferEngine instead of invoking backend
 * copy operations or mutating coherence directly. The interface separates
 * placement-time movement from execution-time residency checks and requires
 * exact producer/consumer streams for every asynchronous GPU boundary.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "backends/DeviceId.h"
#include "tensors/CoherenceState.h"
#include "transfer/TransferMethod.h"

// Forward declarations
namespace llaminar2
{
    class IBackend;
    class IGPUGraphCapture;
    class ITensor;
    class TensorBase;
} // namespace llaminar2

namespace llaminar2
{

    /**
     * @brief Immutable backend-pinned host allocation for captured transfers.
     *
     * The allocation is created only by @ref TransferEngine, is registered for
     * one exact GPU backend/device, and cannot be resized or rebound.  Keeping
     * the registration identity in this type prevents captured H2D/D2H nodes
     * from accepting an arbitrary pageable pointer whose lifetime or backend
     * registration cannot be proven across graph replays.
     *
     * The owner does not synchronize in its destructor.  Its enclosing graph
     * family must destroy/fence every executable and completion event before
     * releasing this allocation.
     */
    class PinnedHostTransferBuffer final
    {
    public:
        /** @brief Release the exact backend-pinned allocation. */
        ~PinnedHostTransferBuffer();

        PinnedHostTransferBuffer(const PinnedHostTransferBuffer &) = delete;
        PinnedHostTransferBuffer &operator=(const PinnedHostTransferBuffer &) = delete;
        PinnedHostTransferBuffer(PinnedHostTransferBuffer &&) = delete;
        PinnedHostTransferBuffer &operator=(PinnedHostTransferBuffer &&) = delete;

        /** @return Total immutable byte capacity. */
        [[nodiscard]] size_t sizeBytes() const noexcept { return bytes_; }

        /** @return GPU whose backend owns the host registration. */
        [[nodiscard]] DeviceId registrationDevice() const noexcept
        {
            return registration_device_;
        }

        /** @return Whether the fixed identity owns its backend allocation. */
        [[nodiscard]] bool isBound() const noexcept
        {
            return allocation_ != nullptr && backend_ != nullptr;
        }

        /** @return Whether bytes, device, and allocation match a capture contract. */
        [[nodiscard]] bool matches(
            size_t bytes,
            DeviceId registration_device) const noexcept
        {
            return isBound() && bytes_ == bytes &&
                   registration_device_ == registration_device;
        }

        /**
         * @brief Return a mutable host address at an exact byte offset.
         * @throws std::out_of_range when @p offset exceeds the allocation.
         */
        [[nodiscard]] void *mutableData(size_t offset = 0) const;

        /**
         * @brief Return a read-only host address at an exact byte offset.
         * @throws std::out_of_range when @p offset exceeds the allocation.
         */
        [[nodiscard]] const void *data(size_t offset = 0) const;

        /**
         * @brief Check that a transfer region belongs to this allocation.
         *
         * The subtraction form deliberately avoids overflow when validating
         * adversarial offsets and byte counts.
         */
        [[nodiscard]] bool contains(size_t offset, size_t bytes) const noexcept
        {
            return offset <= bytes_ && bytes <= bytes_ - offset;
        }

    private:
        friend class TransferEngine;

        /** @brief Exact teardown operation for the stable host address. */
        enum class Ownership : uint8_t
        {
            Unbound,
            BackendAllocation,
            ExternalRegistration,
        };

        /**
         * @brief Declare one immutable transfer identity before backend setup.
         * @throws std::invalid_argument for invalid geometry or device identity.
         */
        PinnedHostTransferBuffer(
            size_t bytes,
            DeviceId registration_device);

        /**
         * @brief Bind the declared identity once through its exact backend.
         * @throws std::logic_error when a different backend tries to rebind it.
         * @throws std::runtime_error when allocation fails.
         */
        void bind(IBackend *backend);

        /**
         * @brief Page-register caller-owned storage without taking allocation ownership.
         *
         * The retained lifetime guarantees that an mmap-backed channel remains
         * mapped until after backend unregistration completes.
         */
        void bindExternal(
            IBackend *backend,
            void *allocation,
            std::shared_ptr<void> lifetime);

        void *allocation_ = nullptr; ///< Stable host address embedded by graph capture.
        size_t bytes_ = 0; ///< Immutable allocation capacity.
        DeviceId registration_device_ = DeviceId::invalid(); ///< Exact registration owner.
        IBackend *backend_ = nullptr; ///< Borrowed process-lifetime backend authority.
        Ownership ownership_ = Ownership::Unbound; ///< Exact teardown contract.
        std::shared_ptr<void> external_lifetime_; ///< Keeps external pages mapped.
    };

    /**
     * @brief Immutable device allocation reserved exclusively for captured DMA.
     *
     * Unlike a tensor, this buffer carries no semantic value or independent
     * coherence generation. It is graph-private transport scratch: one exact
     * stream writes it and later nodes in the same retained transaction consume
     * it. TransferEngine owns allocation, bounds validation, backend identity,
     * and teardown so packet transports never call raw backend allocation/copy
     * APIs or pretend that a bounce buffer is a model tensor.
     *
     * The owner never synchronizes in its destructor. The graph or transport
     * retaining it must prove that every executable and stream is quiescent
     * before releasing the final shared owner.
     */
    class DeviceTransferBuffer final
    {
    public:
        /** @brief Free the exact backend allocation without synchronizing. */
        ~DeviceTransferBuffer();

        DeviceTransferBuffer(const DeviceTransferBuffer &) = delete;
        DeviceTransferBuffer &operator=(const DeviceTransferBuffer &) = delete;
        DeviceTransferBuffer(DeviceTransferBuffer &&) = delete;
        DeviceTransferBuffer &operator=(DeviceTransferBuffer &&) = delete;

        /** @return Immutable allocation capacity in bytes. */
        [[nodiscard]] size_t sizeBytes() const noexcept { return bytes_; }

        /** @return Exact GPU that owns the allocation. */
        [[nodiscard]] DeviceId device() const noexcept { return device_; }

        /** @return Whether setup successfully bound the complete identity. */
        [[nodiscard]] bool isBound() const noexcept
        {
            return allocation_ != nullptr && backend_ != nullptr &&
                   device_.is_gpu() && bytes_ > 0u;
        }

        /** @return Whether an offset/count lies wholly inside the allocation. */
        [[nodiscard]] bool contains(size_t offset, size_t bytes) const noexcept
        {
            return offset <= bytes_ && bytes <= bytes_ - offset;
        }

        /**
         * @brief Return the stable device address at @p offset.
         * @throws std::out_of_range for an unbound or out-of-range access.
         */
        [[nodiscard]] void *mutableDeviceData(size_t offset = 0u) const;

        /**
         * @brief Return the stable read-only device address at @p offset.
         * @throws std::out_of_range for an unbound or out-of-range access.
         */
        [[nodiscard]] const void *deviceData(size_t offset = 0u) const;

    private:
        friend class TransferEngine;

        /** @brief Declare one positive-capacity exact-device identity. */
        DeviceTransferBuffer(size_t bytes, DeviceId device);

        /**
         * @brief Allocate this identity once through its resolved backend.
         * @throws std::logic_error if a second backend attempts to rebind it.
         * @throws std::runtime_error when allocation fails.
         */
        void bind(IBackend *backend);

        void *allocation_ = nullptr; ///< Stable address embedded by graph capture.
        size_t bytes_ = 0u; ///< Immutable byte capacity.
        DeviceId device_ = DeviceId::invalid(); ///< Exact allocation endpoint.
        IBackend *backend_ = nullptr; ///< Borrowed process-lifetime authority.
    };

    /**
     * @brief Unforgeable producer-event publication for one tensor input fork.
     *
     * TransferEngine creates this token only after validating the tensor on its
     * canonical producer stream and recording @ref event_ on that exact stream.
     * It carries no global coherence mutation: consumers must acquire the event
     * on their own explicit streams before any transfer can read the bytes.
     */
    class DeviceTransferInputFork final
    {
    public:
        /** @return Whether the producer tensor, event, device, and stream exist. */
        [[nodiscard]] bool valid() const noexcept
        {
            return source_owner_ && device_.is_gpu() && producer_stream_ &&
                   event_;
        }

        /** @return Exact GPU whose producer stream recorded the event. */
        [[nodiscard]] DeviceId device() const noexcept { return device_; }

    private:
        friend class TransferEngine;

        /** @brief Construct only after the exact producer event is accepted. */
        DeviceTransferInputFork(
            TensorBase *source_owner,
            DeviceId device,
            void *producer_stream,
            void *event) noexcept
            : source_owner_(source_owner),
              device_(device),
              producer_stream_(producer_stream),
              event_(event)
        {
        }

        TensorBase *source_owner_ = nullptr; ///< Canonical transfer-storage owner.
        DeviceId device_ = DeviceId::invalid(); ///< Exact producer GPU.
        void *producer_stream_ = nullptr; ///< Stream validated against tensor authority.
        void *event_ = nullptr; ///< Event recorded after the producer frontier.
    };

    /**
     * @brief Unforgeable proof that a transfer stream acquired a tensor fork.
     *
     * The token embeds the exact consumer stream, so a later DMA submission
     * cannot silently substitute the graph's main stream, a backend default, or
     * another auxiliary lane. It is obtained only through
     * @ref TransferEngine::acquireDeviceInputFork.
     */
    class AcquiredDeviceTransferInput final
    {
    public:
        /** @return Whether source ownership and exact consumer identity exist. */
        [[nodiscard]] bool valid() const noexcept
        {
            return source_owner_ && device_.is_gpu() && consumer_stream_ &&
                   event_;
        }

        /** @return Exact stream on which the subsequent DMA must be enqueued. */
        [[nodiscard]] void *consumerStream() const noexcept
        {
            return consumer_stream_;
        }

    private:
        friend class TransferEngine;

        /** @brief Construct only after the backend accepts the event wait. */
        AcquiredDeviceTransferInput(
            TensorBase *source_owner,
            DeviceId device,
            void *consumer_stream,
            void *event) noexcept
            : source_owner_(source_owner),
              device_(device),
              consumer_stream_(consumer_stream),
              event_(event)
        {
        }

        TensorBase *source_owner_ = nullptr; ///< Canonical transfer-storage owner.
        DeviceId device_ = DeviceId::invalid(); ///< Exact source/consumer GPU.
        void *consumer_stream_ = nullptr; ///< Stream that acquired @ref event_.
        void *event_ = nullptr; ///< Producer frontier acquired by the stream.
    };

    /**
     * @brief One external shared-page region mapped into arbitrary local endpoints.
     *
     * A region may be visible to any mixture of CPU, CUDA, and ROCm devices.
     * GPU pages are registered once per backend family and then resolved to an
     * exact alias for every listed device; CPU aliases are the host address.
     * Device ordering in the list is topology data, never a hard-coded backend
     * role. The owner performs setup-time registration only and never allocates,
     * synchronizes, or changes mapping identity during inference.
     */
    class MappedHostTransferRegion final
    {
    public:
        /** @brief Unregister each backend before releasing the external mapping. */
        ~MappedHostTransferRegion();

        MappedHostTransferRegion(const MappedHostTransferRegion &) = delete;
        MappedHostTransferRegion &operator=(const MappedHostTransferRegion &) = delete;
        MappedHostTransferRegion(MappedHostTransferRegion &&) = delete;
        MappedHostTransferRegion &operator=(MappedHostTransferRegion &&) = delete;

        /** @return Immutable mapped byte capacity. */
        [[nodiscard]] size_t sizeBytes() const noexcept { return bytes_; }

        /** @return Canonical sorted device set registered during setup. */
        [[nodiscard]] std::span<const DeviceId> devices() const noexcept
        {
            return devices_;
        }

        /** @return Whether every declared endpoint owns an exact alias. */
        [[nodiscard]] bool isBound() const noexcept;

        /** @return Whether an offset/count lies wholly inside the region. */
        [[nodiscard]] bool contains(size_t offset, size_t bytes) const noexcept
        {
            return offset <= bytes_ && bytes <= bytes_ - offset;
        }

        /**
         * @brief Return mutable host storage at @p offset.
         * @throws std::out_of_range for an unbound or out-of-range access.
         */
        [[nodiscard]] void *mutableHostData(size_t offset = 0u) const;

        /**
         * @brief Return one endpoint's device-visible alias at @p offset.
         * @throws std::invalid_argument when the device was not registered.
         * @throws std::out_of_range for an invalid offset.
         */
        [[nodiscard]] void *deviceAlias(
            DeviceId device,
            size_t offset = 0u) const;

        /** @return Whether the exact device was declared and successfully bound. */
        [[nodiscard]] bool hasDevice(DeviceId device) const noexcept;

    private:
        friend class TransferEngine;

        /** One exact endpoint alias and backend ownership witness. */
        struct DeviceAlias
        {
            DeviceId device = DeviceId::invalid();
            void *address = nullptr;
            IBackend *backend = nullptr; ///< Null only for a CPU alias.
        };

        /** One portable registration shared by all same-family device aliases. */
        struct BackendRegistration
        {
            DeviceType type = DeviceType::CPU;
            IBackend *backend = nullptr;
            int registration_ordinal = -1;
        };

        /** @brief Validate and retain immutable external mapping identity. */
        MappedHostTransferRegion(
            void *allocation,
            size_t bytes,
            std::span<const DeviceId> devices,
            std::shared_ptr<void> lifetime);

        /** @return Backend bound to @p device, or null for CPU/absent. */
        [[nodiscard]] IBackend *backendFor(DeviceId device) const noexcept;

        void *allocation_ = nullptr; ///< Stable process-local mapping address.
        size_t bytes_ = 0u; ///< Immutable complete region size.
        std::vector<DeviceId> devices_; ///< Canonical exact endpoint set.
        std::vector<DeviceAlias> aliases_; ///< One alias per @ref devices_.
        std::vector<BackendRegistration> registrations_; ///< One per GPU family.
        std::shared_ptr<void> external_lifetime_; ///< Unmapped only after unregister.
        bool bound_ = false; ///< True after every alias resolves successfully.
    };

    /**
     * @brief Unforgeable setup binding for a mapped timeline acquire in a kernel.
     *
     * Only TransferEngine can construct a valid instance, after proving mapped
     * region bounds, exact endpoint registration, backend ownership, alignment,
     * and system-scope timeline support. Packet stages may expose the address
     * only to their backend kernel launch; ordinary production callers should
     * use the stream-enqueue or retained-transaction APIs instead.
     */
    class MappedTimelineKernelWait64Binding final
    {
    public:
        /** @return Whether TransferEngine resolved a complete immutable edge. */
        [[nodiscard]] bool valid() const noexcept
        {
            return signal_ && value_ != 0u && device_.is_gpu();
        }

        /** @return Exact read-only device alias embedded by graph capture. */
        [[nodiscard]] const std::uint64_t *deviceSignal() const noexcept
        {
            return signal_;
        }

        /** @return Positive unsigned-GEQ lease value awaited by the kernel. */
        [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

        /** @return Exact endpoint whose address space interprets the alias. */
        [[nodiscard]] DeviceId device() const noexcept { return device_; }

    private:
        friend class TransferEngine;

        /** @brief Construct only after TransferEngine validates the full edge. */
        MappedTimelineKernelWait64Binding(
            const std::uint64_t *signal,
            std::uint64_t value,
            DeviceId device) noexcept
            : signal_(signal), value_(value), device_(device)
        {
        }

        const std::uint64_t *signal_ = nullptr; ///< Stable mapped device alias.
        std::uint64_t value_ = 0u; ///< Immutable positive lease value.
        DeviceId device_ = DeviceId::invalid(); ///< Exact alias address space.
    };

    /**
     * @brief Unforgeable setup binding for a mapped timeline release in a kernel.
     *
     * This type is deliberately distinct from the acquire binding: a caller
     * cannot accidentally pass a read edge to a publishing kernel or mutate a
     * peer-owned wait address. TransferEngine remains the sole alias authority.
     */
    class MappedTimelineKernelPublish64Binding final
    {
    public:
        /** @return Whether TransferEngine resolved a complete immutable edge. */
        [[nodiscard]] bool valid() const noexcept
        {
            return signal_ && value_ != 0u && device_.is_gpu();
        }

        /** @return Exact writable device alias embedded by graph capture. */
        [[nodiscard]] std::uint64_t *deviceSignal() const noexcept
        {
            return signal_;
        }

        /** @return Positive monotonic lease value published by the kernel. */
        [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

        /** @return Exact endpoint whose address space interprets the alias. */
        [[nodiscard]] DeviceId device() const noexcept { return device_; }

    private:
        friend class TransferEngine;

        /** @brief Construct only after TransferEngine validates the full edge. */
        MappedTimelineKernelPublish64Binding(
            std::uint64_t *signal,
            std::uint64_t value,
            DeviceId device) noexcept
            : signal_(signal), value_(value), device_(device)
        {
        }

        std::uint64_t *signal_ = nullptr; ///< Stable mapped device alias.
        std::uint64_t value_ = 0u; ///< Immutable positive lease value.
        DeviceId device_ = DeviceId::invalid(); ///< Exact alias address space.
    };

    /** @brief One ordinary captured fragment in a mapped-timeline transaction. */
    struct MappedTimelineCapturedFragment
    {
        const char *name = nullptr; ///< Stable semantic role for diagnostics.
        const IGPUGraphCapture *capture = nullptr; ///< Borrowed same-device graph.
    };

    /** @brief One native unsigned-GEQ mapped 64-bit wait graph node. */
    struct MappedTimelineWait64
    {
        const char *name = nullptr; ///< Stable semantic edge identity.
        const MappedHostTransferRegion *region = nullptr; ///< Registered page owner.
        size_t signal_offset = 0u; ///< Aligned word relative to @ref region.
        std::uint64_t value = 0u; ///< Positive capture-stable lease value.
    };

    /** @brief One native fenced mapped 64-bit publication graph node. */
    struct MappedTimelinePublish64
    {
        const char *name = nullptr; ///< Stable semantic edge identity.
        const MappedHostTransferRegion *region = nullptr; ///< Registered page owner.
        size_t signal_offset = 0u; ///< Aligned word relative to @ref region.
        std::uint64_t value = 0u; ///< Positive capture-stable lease value.
    };

    /**
     * @brief Typed step in one retained graph-native mapped timeline transaction.
     *
     * A variant prevents callers from combining a fragment pointer with a
     * signal operation or accidentally selecting wait/publication through a
     * boolean. TransferEngine resolves every mapped alias before lowering.
     */
    using MappedTimelineTransactionStep = std::variant<
        MappedTimelineCapturedFragment,
        MappedTimelineWait64,
        MappedTimelinePublish64>;

    /// TransferEngine — Unified, testable data movement between host and devices.
    ///
    /// This is the sole public authority for tensor movement and coherence
    /// publication. All state lives in TensorBase and backend-owned resources;
    /// callers describe completed work here instead of mutating TensorBase's
    /// coherence state directly.
    ///
    /// Usage for coherence:
    ///   TransferEngine engine;
    ///   engine.upload(tensor, DeviceId::cuda(0));    // ensureOnDevice replacement
    ///   engine.download(tensor);                      // ensureOnHost replacement
    ///
    /// Usage for PP activation transfer:
    ///   engine.transferActivation(tensor, next_stage_device);
    ///
    /// Canonical decision tree (planTransfer):
    ///   src == dst                       → NOOP
    ///   residency == MAPPED              → MAPPED_NOOP
    ///   CPU → GPU                        → HOST_TO_DEVICE
    ///   GPU → CPU                        → DEVICE_TO_HOST
    ///   GPU → GPU same vendor            → DEVICE_TO_DEVICE_SAME_BACKEND
    ///   GPU → GPU cross-vendor           → HOST_STAGED
    ///
    class TransferEngine
    {
    public:
        TransferEngine() = default;

        /// Construct a TransferEngine with an injected backend resolver
        /// (for testing with MockBackend).
        using BackendResolver = IBackend *(*)(DeviceId);
        explicit TransferEngine(BackendResolver resolver) : resolve_(resolver) {}

        /**
         * @brief Declare an immutable pinned-buffer identity without GPU work.
         *
         * Device-free graph-policy tests and model graph construction can own
         * this token before a backend context exists.  Production setup must
         * call @ref bindPinnedHostBuffer before capture; copy APIs reject an
         * unbound token.
         */
        [[nodiscard]] std::shared_ptr<PinnedHostTransferBuffer>
        declarePinnedHostBuffer(
            size_t bytes,
            DeviceId registration_device) const;

        /**
         * @brief Idempotently allocate a declared pinned-buffer identity.
         * @throws std::invalid_argument when identity is not GPU-valid.
         * @throws std::runtime_error for missing backend/allocation failure.
         */
        void bindPinnedHostBuffer(
            PinnedHostTransferBuffer &buffer) const;

        /**
         * @brief Allocate a model-lifetime host buffer for captured GPU DMA.
         *
         * Unlike tensor host pinning, this allocation is independent of tensor
         * coherence.  A host protocol may overwrite it between serial graph
         * replays without downloading the preceding device tensor generation.
         *
         * @param bytes Positive immutable capacity.
         * @param registration_device Exact CUDA/ROCm device/backend owner.
         * @return Shared RAII owner suitable for retention by graph families.
         * @throws std::invalid_argument for zero bytes or a non-GPU device.
         * @throws std::runtime_error when no backend/allocation is available.
         */
        [[nodiscard]] std::shared_ptr<PinnedHostTransferBuffer>
        allocatePinnedHostBuffer(
            size_t bytes,
            DeviceId registration_device) const;

        /**
         * @brief Allocate graph-private persistent device transport scratch.
         *
         * This setup-only operation is the device counterpart to
         * @ref allocatePinnedHostBuffer. The returned address and capacity never
         * change, so captured DMA and packet kernels may retain them. The buffer
         * has no tensor coherence state; ordering must remain entirely within the
         * exact retained stream/graph transaction.
         *
         * @param bytes Positive immutable allocation capacity.
         * @param device Exact CUDA/ROCm endpoint owning the bytes.
         * @return Shared RAII owner suitable for graph-family retention.
         */
        [[nodiscard]] std::shared_ptr<DeviceTransferBuffer>
        allocateDeviceTransferBuffer(size_t bytes, DeviceId device) const;

        /**
         * @brief Register an immutable caller-owned host region for captured DMA.
         *
         * This is the canonical bridge from a model-lifetime shared-memory
         * channel to the event-ordered pinned transfer APIs. No allocation or
         * payload copy occurs here.
         *
         * @param allocation Stable non-null host address.
         * @param bytes Positive immutable byte capacity.
         * @param registration_device Exact GPU/backend capturing the copies.
         * @param lifetime Owner that keeps the address mapped through unregister.
         */
        [[nodiscard]] std::shared_ptr<PinnedHostTransferBuffer>
        registerExternalPinnedHostBuffer(
            void *allocation,
            size_t bytes,
            DeviceId registration_device,
            std::shared_ptr<void> lifetime) const;

        /**
         * @brief Register shared external pages for arbitrary local endpoints.
         *
         * Devices may contain CPU, CUDA, ROCm, or any combination in any order.
         * GPU registration is portable within a backend family and produces an
         * exact alias per ordinal. Duplicate/invalid devices and partial backend
         * registration fail atomically; successful teardown unregisters every
         * family before releasing @p lifetime.
         */
        [[nodiscard]] std::shared_ptr<MappedHostTransferRegion>
        registerExternalMappedHostRegion(
            void *allocation,
            size_t bytes,
            std::span<const DeviceId> devices,
            std::shared_ptr<void> lifetime) const;

        /**
         * @brief Bind one mapped acquire edge for a fused packet kernel.
         *
         * This setup-only method resolves an exact stable device alias without
         * enqueueing work. It is intentionally narrower than exposing
         * MappedHostTransferRegion::deviceAlias to packet code: the returned
         * type proves bounds, alignment, backend ownership, endpoint identity,
         * positive value, and native system-scope timeline capability.
         */
        [[nodiscard]] MappedTimelineKernelWait64Binding
        bindMappedTimelineKernelWait64(
            const MappedHostTransferRegion &region,
            size_t signal_offset,
            std::uint64_t value,
            DeviceId device) const;

        /**
         * @brief Bind one mapped release edge for a fused packet kernel.
         *
         * The binding performs no stream operation or synchronization. Its
         * immutable address/value pair becomes complete graph capture identity
         * for a backend packet kernel that issues the final system-release store.
         */
        [[nodiscard]] MappedTimelineKernelPublish64Binding
        bindMappedTimelineKernelPublish64(
            const MappedHostTransferRegion &region,
            size_t signal_offset,
            std::uint64_t value,
            DeviceId device) const;

        /**
         * @brief Enqueue one 64-bit mapped timeline wait on an exact GPU stream.
         * @param region Canonically registered shared pages.
         * @param signal_offset Aligned byte offset of the 64-bit signal word.
         * @param value Positive monotonic value to await using unsigned GEQ.
         * @param device Exact local GPU interpreting the region alias.
         * @param stream Exact non-null consumer stream.
         */
        void enqueueMappedTimelineWait64(
            const MappedHostTransferRegion &region,
            size_t signal_offset,
            std::uint64_t value,
            DeviceId device,
            void *stream) const;

        /**
         * @brief Enqueue one fenced 64-bit mapped timeline publication.
         *
         * Publication occurs after all preceding packet writes/copies on the
         * same stream and never waits or synchronizes the host.
         */
        void enqueueMappedTimelinePublish64(
            const MappedHostTransferRegion &region,
            size_t signal_offset,
            std::uint64_t value,
            DeviceId device,
            void *stream) const;

        /**
         * @brief Enqueue registered shared-page bytes into stable tensor storage.
         *
         * This is the bulk-payload companion to @ref enqueueMappedTimelineWait64.
         * The caller first orders @p stream after the producer's mapped timeline
         * word, then calls this method to let the backend DMA engine read the
         * already registered host pages. No registration, allocation, host wait,
         * or stream synchronization occurs here, so the copy is safe to retain
         * as a node in a captured transaction.
         *
         * @param region Canonically registered shared-page owner.
         * @param source_offset Byte offset within @p region.
         * @param destination Tensor with preallocated storage on @p device.
         * @param destination_offset Byte offset within @p destination.
         * @param bytes Positive byte count contained by both regions.
         * @param device Exact GPU whose backend registered the shared pages.
         * @param stream Exact non-null consumer stream.
         * @throws std::invalid_argument or std::out_of_range for bad geometry.
         * @throws std::runtime_error for residency/backend/enqueue failures.
         */
        void enqueueMappedHostToDevice(
            const MappedHostTransferRegion &region,
            size_t source_offset,
            ITensor *destination,
            size_t destination_offset,
            size_t bytes,
            DeviceId device,
            void *stream) const;

        /**
         * @brief Enqueue mapped shared pages into graph-private device scratch.
         *
         * No tensor publication is needed or permitted: @p destination is a
         * transport-local buffer whose producer and consumers are ordered by the
         * enclosing exact stream. The method performs no allocation, host wait,
         * registration, or synchronization and is safe during native capture.
         */
        void enqueueMappedHostToDevice(
            const MappedHostTransferRegion &region,
            size_t source_offset,
            DeviceTransferBuffer &destination,
            size_t destination_offset,
            size_t bytes,
            DeviceId device,
            void *stream) const;

        /**
         * @brief Publish one tensor producer frontier for auxiliary transfer streams.
         *
         * The tensor is validated against @p producer_stream using the ordinary
         * graph dependency ledger. TransferEngine then records @p event on that
         * same stream and returns a token that can be acquired by any number of
         * same-device transfer lanes. No host wait, synchronization, allocation,
         * or tensor-authority mutation occurs.
         *
         * @param source Tensor read by the later asynchronous transfer.
         * @param device Exact GPU owning both producer and consumer streams.
         * @param producer_stream Exact tensor-producing graph stream.
         * @param event Setup-owned event retained with the graph transaction.
         * @return Unforgeable producer-event publication token.
         * @throws std::invalid_argument for incomplete identity.
         * @throws std::runtime_error when residency or event recording fails.
         */
        [[nodiscard]] DeviceTransferInputFork recordDeviceInputFork(
            ITensor *source,
            DeviceId device,
            void *producer_stream,
            void *event) const;

        /**
         * @brief Acquire a published tensor frontier on one auxiliary stream.
         *
         * The backend wait is enqueued without blocking the host. Successful
         * return is the only contract accepted by the fork-aware DMA overload;
         * callers cannot pass the producer token directly or choose a different
         * stream after acquisition.
         *
         * @param publication Producer token returned by recordDeviceInputFork().
         * @param consumer_stream Exact non-null auxiliary transfer stream.
         * @return Unforgeable input token bound to @p consumer_stream.
         * @throws std::invalid_argument for incomplete or mismatched identity.
         * @throws std::runtime_error when the backend rejects the event wait.
         */
        [[nodiscard]] AcquiredDeviceTransferInput acquireDeviceInputFork(
            const DeviceTransferInputFork &publication,
            void *consumer_stream) const;

        /**
         * @brief Enqueue event-acquired tensor bytes into mapped shared pages.
         *
         * Unlike the ordinary tensor overload, this method does not attempt a
         * second coherence join on the auxiliary stream. @p source already
         * proves that its embedded stream acquired the canonical producer event;
         * that same stream is used for the D2H submission and is not separately
         * supplied by the caller.
         *
         * @param source Event-acquired tensor/stream authority.
         * @param source_offset Byte offset within the tensor owner.
         * @param region Canonically registered shared-page destination.
         * @param destination_offset Byte offset within @p region.
         * @param bytes Positive byte count contained by both storage owners.
         * @throws std::invalid_argument, std::out_of_range, or std::runtime_error
         *         when identity, bounds, backend, or enqueue validation fails.
         */
        void enqueueDeviceToMappedHost(
            const AcquiredDeviceTransferInput &source,
            size_t source_offset,
            const MappedHostTransferRegion &region,
            size_t destination_offset,
            size_t bytes) const;

        /**
         * @brief Enqueue stable tensor bytes into registered shared pages.
         *
         * This is the producer-side bulk-payload companion to
         * @ref enqueueMappedTimelinePublish64. The caller publishes the mapped
         * timeline only after this copy on the same stream, which gives a remote
         * heterogeneous consumer one release/acquire edge covering every payload
         * byte. The method never waits for host visibility or synchronizes.
         *
         * @param source Tensor resident on @p device.
         * @param source_offset Byte offset within @p source.
         * @param region Canonically registered shared-page destination owner.
         * @param destination_offset Byte offset within @p region.
         * @param bytes Positive byte count contained by both regions.
         * @param device Exact source GPU and registration identity.
         * @param stream Exact non-null producer stream.
         * @throws std::invalid_argument or std::out_of_range for bad geometry.
         * @throws std::runtime_error for residency/backend/enqueue failures.
         */
        void enqueueDeviceToMappedHost(
            ITensor *source,
            size_t source_offset,
            const MappedHostTransferRegion &region,
            size_t destination_offset,
            size_t bytes,
            DeviceId device,
            void *stream) const;

        /**
         * @brief Enqueue graph-private device scratch into mapped shared pages.
         *
         * The caller publishes a mapped timeline word only after this operation
         * on the same exact stream. Bounds and backend ownership are checked here;
         * no tensor coherence, event, allocation, wait, or synchronization occurs.
         */
        void enqueueDeviceToMappedHost(
            const DeviceTransferBuffer &source,
            size_t source_offset,
            const MappedHostTransferRegion &region,
            size_t destination_offset,
            size_t bytes,
            DeviceId device,
            void *stream) const;

        /**
         * @brief Build a retained graph from fragments and native timeline nodes.
         *
         * This setup-only operation is the graph counterpart of the direct
         * enqueue APIs. It resolves exact aliases through registered regions,
         * validates device/backend ownership, and asks the graph backend to
         * create explicit device-owned timeline nodes. Scalar stream wait/write
         * calls are never captured because their retention semantics differ by
         * backend.
         *
         * @param destination Empty graph owner for the exact @p device.
         * @param ordered_steps Non-empty producer-to-consumer transaction.
         * @param device Planner-selected GPU owning every fragment/node.
         * @throws std::invalid_argument for incomplete or mismatched bindings.
         * @throws std::runtime_error when native graph lowering fails.
         */
        void buildMappedTimelineTransaction(
            IGPUGraphCapture &destination,
            std::span<const MappedTimelineTransactionStep> ordered_steps,
            DeviceId device) const;

        /**
         * @brief Enqueue a fixed pinned-host region into stable tensor storage.
         *
         * The copy uses @p stream exactly and never allocates, waits, or
         * synchronizes.  During graph capture the destination publication is
         * recorded in the capture dependency ledger; outside capture it gains
         * the ordinary exact producer event.  The caller must retain
         * @p pinned_source until the graph/batch completion event has fired.
         *
         * @param pinned_source Typed backend-pinned source owner.
         * @param source_offset Byte offset within @p pinned_source.
         * @param destination Tensor with preallocated storage on @p device.
         * @param destination_offset Byte offset within @p destination.
         * @param bytes Positive byte count contained by both regions.
         * @param device Exact GPU on which the destination resides.
         * @param stream Exact non-null copy/consumer stream.
         * @throws std::invalid_argument or std::out_of_range for bad geometry.
         * @throws std::runtime_error for residency/backend/enqueue failures.
         */
        void enqueuePinnedHostToDevice(
            const PinnedHostTransferBuffer &pinned_source,
            size_t source_offset,
            ITensor *destination,
            size_t destination_offset,
            size_t bytes,
            DeviceId device,
            void *stream) const;

        /**
         * @brief Enqueue stable tensor bytes into a fixed pinned-host region.
         *
         * This is the D2H companion to @ref enqueuePinnedHostToDevice.  It
         * joins the tensor producer to @p stream, submits one asynchronous
         * copy, and leaves host visibility to the enclosing graph/batch event.
         * It never waits or synchronizes.
         *
         * @param source Tensor resident on @p device.
         * @param source_offset Byte offset within @p source.
         * @param pinned_destination Typed backend-pinned destination owner.
         * @param destination_offset Byte offset within the pinned allocation.
         * @param bytes Positive byte count contained by both regions.
         * @param device Exact source GPU and registration identity.
         * @param stream Exact non-null producer/copy stream.
         * @throws std::invalid_argument or std::out_of_range for bad geometry.
         * @throws std::runtime_error for residency/backend/enqueue failures.
         */
        void enqueueDeviceToPinnedHost(
            ITensor *source,
            size_t source_offset,
            const PinnedHostTransferBuffer &pinned_destination,
            size_t destination_offset,
            size_t bytes,
            DeviceId device,
            void *stream) const;

        // ========================================================================
        // Plan: pure logic — determines HOW to transfer. No side effects.
        // ========================================================================

        /// Determine the transfer method for moving data between devices.
        /// This is a pure function: no state, no side effects, fully testable.
        static TransferMethod planTransfer(DeviceId src, DeviceId dst, MemoryResidency residency);

        /// Human-readable description of a transfer plan (for --explain-placement).
        static std::string describeTransferPlan(DeviceId src, DeviceId dst, MemoryResidency residency);

        // ========================================================================
        // Execute: actually moves data.
        // ========================================================================

        /// Execute a planned transfer using the given request.
        /// Dispatches to the correct IBackend based on TransferMethod.
        TransferResult execute(const TransferRequest &request);

        // ========================================================================
        // High-level API: operates on TensorBase directly.
        // ========================================================================

        /// Upload tensor data from host to target device.
        /// Updates TensorCoherenceState appropriately.
        TransferResult upload(TensorBase *tensor, DeviceId target_device);

        /// Download tensor data from device to host.
        /// Updates TensorCoherenceState appropriately.
        TransferResult download(TensorBase *tensor);

        /**
         * @brief Move an activation tensor to a pipeline stage's device.
         *
         * The service selects the authoritative source, performs the required
         * transport, promotes the destination storage into the tensor's primary
         * device slot, and publishes destination ownership. Callers must not
         * repair or restate coherence after this method returns.
         *
         * @param tensor Tensor whose activation storage is being moved.
         * @param target_device Destination CPU or GPU.
         * @param bytes_override Optional active prefix in bytes. Zero moves the
         *        tensor's complete allocation.
         */
        TransferResult transferActivation(
            TensorBase *tensor,
            DeviceId target_device,
            size_t bytes_override = 0);

        /// Copy activation data from one tensor INTO another tensor's buffer on
        /// dst_device, choosing the optimal transport automatically:
        ///   - same physical GPU            → device-to-device copy (intra-VRAM)
        ///   - same-vendor, different GPU   → peer copy (NCCL/RCCL or peer DMA)
        ///   - cross-vendor GPU (CUDA↔ROCm) → host-staged bounce (no direct path)
        ///   - source on host / mapped       → direct H2D (or host memcpy for mapped)
        /// On success, dst becomes authoritative on dst_device.
        ///
        /// Unlike transferActivation() (which moves a single tensor between
        /// devices), this performs a tensor→tensor copy where source and
        /// destination are distinct buffers. Used for the PP hidden-state
        /// handoff, where the producer's output must land in the consumer
        /// graph's working buffer without a redundant device→host→device trip.
        TransferResult copyActivation(TensorBase *src, TensorBase *dst,
                                      DeviceId dst_device, size_t bytes);

        /// Full upload lifecycle: replaces TensorBase::ensureOnDevice() logic.
        /// Handles mapped storage, event dependencies, migration, allocation,
        /// and an asynchronously queued H2D copy. The exact copy-completion
        /// event remains attached to the tensor so GPU consumers wait on-device
        /// and host storage cannot be overwritten or released while DMA reads it.
        /// Caller must hold coherence_mutex_ and check graph-capture/CPU bailouts.
        TransferResult uploadFull(TensorBase *tensor, DeviceId target_device, void *stream = nullptr);

        /// Full download lifecycle: replaces TensorBase::ensureOnHost() logic.
        /// Handles mapped sync, event wait, standard D2H.
        /// Caller must hold coherence_mutex_.
        TransferResult downloadFull(TensorBase *tensor, void *stream = nullptr);

        /**
         * @brief Prepare an input tensor for an asynchronous GPU consumer.
         *
         * This is the public ITensor boundary for upload/residency work.
         * TransferEngine performs the checked conversion to TensorBase, moves
         * the current contents to @p target_device on @p stream, and verifies
         * that the resulting allocation belongs to that exact device.
         *
         * @throws std::invalid_argument for a null tensor or non-GPU target.
         * @throws std::runtime_error if placement, upload, or validation fails.
         */
        static void prepareDeviceInput(
            ITensor *tensor,
            DeviceId target_device,
            void *stream);

        /**
         * @brief Join an already-resident GPU input to its consumer stream.
         *
         * This is the execution-time input contract. The tensor must already
         * own valid storage on exactly @p target_device. The method joins the
         * tensor's published producer event to @p consumer_stream, but it never
         * allocates, uploads, migrates, copies, downloads, or changes coherence
         * authority. Missing residency is therefore a lifecycle failure rather
         * than permission to repair state in the inference hot path.
         *
         * Placement-capable owners must call prepareDeviceInput() before graph
         * execution. Compute stages and graph executors must call this method.
         * Outside graph capture, a successful event join records the exact
         * `{completion event, consumer stream}` pair on the tensor. During graph
         * capture, the method never asks the backend to import an external
         * event: the pair must already match a pre-capture join or execution
         * fails immediately.
         *
         * @param tensor Tensor whose current device bytes will be consumed.
         * @param target_device Exact GPU on which the consumer will execute.
         * @param consumer_stream Exact non-null stream for the consuming work.
         *
         * @throws std::invalid_argument for a null tensor, non-GPU target, or
         *         null consumer stream.
         * @throws std::runtime_error when exact valid residency is absent, the
         *         producer event cannot be joined, or graph capture encounters
         *         an input that was not joined to its exact stream beforehand.
         */
        static void requireDeviceInput(
            ITensor *tensor,
            DeviceId target_device,
            void *consumer_stream);

        /**
         * @brief Prepare an input tensor for a synchronous host consumer.
         *
         * This is the CPU counterpart to prepareDeviceInput(). TransferEngine
         * materializes the current authoritative bytes in host storage and
         * rejects tensors whose device publication cannot be downloaded.
         * Keeping this operation here prevents graph executors from reaching
         * into TensorBase coherence transitions or treating CPU as a degenerate
         * GPU target.
         *
         * @throws std::invalid_argument for a null or unsupported tensor.
         * @throws std::runtime_error when host materialization fails.
         */
        static void prepareHostInput(ITensor *tensor);

        /**
         * @brief Allocate GPU storage without declaring a producer or authority.
         *
         * This operation exists for graph and arena construction, where stable
         * device addresses must be established before any executable stage or
         * producer stream exists. It performs no copy, records no event, and
         * does not publish the device allocation as authoritative.
         *
         * Runtime kernels must use requireDeviceOutput() with their exact
         * producer stream. Keeping allocation separate prevents initialization
         * code from inventing a stream identity merely to obtain a pointer.
         *
         * @throws std::invalid_argument for a null tensor or non-GPU target.
         * @throws std::runtime_error if allocation or validation fails.
         */
        static void allocateDeviceStorage(
            ITensor *tensor,
            DeviceId target_device);

        /**
         * @brief Prepare storage for an output-only asynchronous GPU writer.
         *
         * Unlike prepareDeviceInput(), this never uploads stale host bytes.
         * Placement/admission code may use it to allocate or retarget storage
         * on @p target_device before execution. Captured and hot-path kernels
         * must instead use requireDeviceOutput(), which cannot allocate. The
         * exact non-null @p stream identifies the future producer.
         *
         * @throws std::invalid_argument for a null tensor, non-GPU target, or
         *         null producer stream.
         * @throws std::runtime_error if allocation or validation fails.
         */
        static void prepareDeviceOutput(
            ITensor *tensor,
            DeviceId target_device,
            void *stream);

        /**
         * @brief Require pre-existing output storage on one exact GPU.
         *
         * This is the execution-time output counterpart to
         * requireDeviceInput(). The tensor must already own an allocation on
         * @p target_device, but its bytes need not be valid because the caller
         * is about to overwrite them. The method never allocates, migrates,
         * copies, publishes authority, or records an event.
         *
         * @param tensor Tensor whose storage will receive a GPU write.
         * @param target_device Exact GPU on which the writer will execute.
         * @param producer_stream Exact non-null stream for the upcoming write.
         *
         * @throws std::invalid_argument for a null tensor, non-GPU target, or
         *         null producer stream.
         * @throws std::runtime_error when stable storage is absent from the
         *         exact target device.
         */
        static void requireDeviceOutput(
            ITensor *tensor,
            DeviceId target_device,
            void *producer_stream);

        // ========================================================================
        // Publication: expose completed work without exposing raw state mutation.
        // ========================================================================

        /**
         * @brief Publish a completed or enqueued GPU write.
         *
         * Records a completion event on @p producer_stream and makes the device
         * copy authoritative only after event publication succeeds. During graph
         * capture, the graph controller owns the externally visible completion
         * event and this call validates residency before deferring publication.
         *
         * @param tensor Tensor whose device storage was written.
         * @param device GPU that owns the authoritative storage.
         * @param producer_stream Exact non-null stream that enqueued the write.
         *        A null stream is never interpreted as a backend default because
         *        doing so would sever publication from the actual producer.
         *
         * @throws std::invalid_argument for a null tensor, non-GPU device, or
         *         null producer stream.
         * @throws std::runtime_error when event publication fails.
         */
        static void publishDeviceWrite(
            TensorBase *tensor,
            DeviceId device,
            void *producer_stream);

        /**
         * @brief Publish a GPU write through the public tensor abstraction.
         *
         * TransferEngine owns the checked conversion to the coherence-aware
         * TensorBase implementation. This overload keeps graph and collective
         * callers from coupling themselves to TensorBase solely to publish
         * ordering.
         */
        static void publishDeviceWrite(
            ITensor *tensor,
            DeviceId device,
            void *producer_stream);

        /// Smart-pointer convenience preserving the same publication contract.
        template <typename SmartPointer>
        static auto publishDeviceWrite(
            const SmartPointer &tensor,
            DeviceId device,
            void *producer_stream)
            -> decltype(tensor.get(), void())
        {
            publishDeviceWrite(tensor.get(), device, producer_stream);
        }

        /**
         * @brief Publish a GPU write after a blocking operation has completed.
         *
         * This is the only streamless GPU publication contract. It is reserved
         * for transfer implementations whose backend call guarantees that all
         * written bytes are complete before returning. There is no outstanding
         * producer to represent with an event, so the method retires stale
         * completion metadata and publishes device authority directly.
         *
         * Kernel launches, asynchronous copies, and collectives must never use
         * this method. They must call publishDeviceWrite() with the exact
         * non-null stream on which their write was enqueued.
         *
         * @param tensor Tensor whose device storage was written.
         * @param device GPU that owns the completed destination storage.
         *
         * @throws std::invalid_argument for a null tensor or non-GPU device.
         * @throws std::runtime_error when tensor storage does not belong to
         *         @p device.
         */
        static void publishCompletedDeviceWrite(
            TensorBase *tensor,
            DeviceId device);

        /// ITensor overload; see the TensorBase overload for the contract.
        static void publishCompletedDeviceWrite(
            ITensor *tensor,
            DeviceId device);

        /// Smart-pointer convenience preserving the same publication contract.
        template <typename SmartPointer>
        static auto publishCompletedDeviceWrite(
            const SmartPointer &tensor,
            DeviceId device)
            -> decltype(tensor.get(), void())
        {
            publishCompletedDeviceWrite(tensor.get(), device);
        }

        /**
         * @brief Publish a GPU write using the tensor's current device.
         *
         * This convenience is intended for isolated kernel harnesses whose
         * tensor allocation already fixes the device unambiguously. Production
         * orchestration should prefer the explicit-device overload so placement
         * errors cannot be hidden by ambient tensor state.
         *
         * @throws std::invalid_argument when @p producer_stream is null.
         * @throws std::runtime_error when the tensor has no current GPU device.
         */
        static void publishCurrentDeviceWrite(
            TensorBase *tensor,
            void *producer_stream);

        /// ITensor overload; see the TensorBase overload for the contract.
        static void publishCurrentDeviceWrite(
            ITensor *tensor,
            void *producer_stream);

        /// Smart-pointer convenience preserving the same publication contract.
        template <typename SmartPointer>
        static auto publishCurrentDeviceWrite(
            const SmartPointer &tensor,
            void *producer_stream)
            -> decltype(tensor.get(), void())
        {
            publishCurrentDeviceWrite(tensor.get(), producer_stream);
        }

        /**
         * @brief Publish a graph-replay write whose completion event is graph-owned.
         *
         * This deliberately performs no per-tensor event record. It is reserved
         * for BufferArena/graph-controller code that has already attached the
         * replay's graph-level completion event to every external consumer.
         *
         * @throws std::invalid_argument for a null tensor or non-GPU device.
         */
        static void publishGraphOwnedDeviceWrite(
            TensorBase *tensor,
            DeviceId device);

        /// ITensor overload; see the TensorBase overload for the contract.
        static void publishGraphOwnedDeviceWrite(
            ITensor *tensor,
            DeviceId device);

        /// Smart-pointer convenience preserving the same publication contract.
        template <typename SmartPointer>
        static auto publishGraphOwnedDeviceWrite(
            const SmartPointer &tensor,
            DeviceId device)
            -> decltype(tensor.get(), void())
        {
            publishGraphOwnedDeviceWrite(tensor.get(), device);
        }

        /**
         * @brief Publish a graph-owned write using current tensor placement.
         *
         * This is the flags-only counterpart to
         * publishCurrentDeviceWrite(). It exists for graph-controller adapters
         * and coherence state-machine tests; ordinary asynchronous producers
         * must publish an event-backed write instead.
         */
        static void publishGraphOwnedCurrentDeviceWrite(TensorBase *tensor);

        /// ITensor overload; see the TensorBase overload for the contract.
        static void publishGraphOwnedCurrentDeviceWrite(ITensor *tensor);

        /// Smart-pointer convenience preserving the same publication contract.
        template <typename SmartPointer>
        static auto publishGraphOwnedCurrentDeviceWrite(
            const SmartPointer &tensor)
            -> decltype(tensor.get(), void())
        {
            publishGraphOwnedCurrentDeviceWrite(tensor.get());
        }

        /**
         * @brief Publish host memory after an external host writer completes.
         *
         * Mapped tensors remain mapped; ordinary tensors make their existing
         * device allocation stale without moving bytes.
         */
        static void publishHostWrite(TensorBase *tensor);

        /// ITensor overload; see the TensorBase overload for the contract.
        static void publishHostWrite(ITensor *tensor);

        /// Smart-pointer convenience preserving the same publication contract.
        template <typename SmartPointer>
        static auto publishHostWrite(const SmartPointer &tensor)
            -> decltype(tensor.get(), void())
        {
            publishHostWrite(tensor.get());
        }

        /**
         * @brief Publish that host and device copies contain identical bytes.
         *
         * This is intended for transfer implementations after a completed copy,
         * not for kernel writers. The method keeps this exceptional state change
         * inside the transfer subsystem.
         */
        static void publishSynchronized(TensorBase *tensor);

        /// ITensor overload; see the TensorBase overload for the contract.
        static void publishSynchronized(ITensor *tensor);

        /// Smart-pointer convenience preserving the same publication contract.
        template <typename SmartPointer>
        static auto publishSynchronized(const SmartPointer &tensor)
            -> decltype(tensor.get(), void())
        {
            publishSynchronized(tensor.get());
        }

        // ========================================================================
        // Singleton access (using default backend resolver)
        // ========================================================================

        /// Get the default TransferEngine instance.
        static TransferEngine &instance();

    private:
        friend class TensorBase;

        /**
         * @brief Wait for and retire a queued H2D use of tensor host storage.
         *
         * Caller must hold TensorBase::coherence_mutex_, or own the tensor
         * exclusively during destruction. Device-write events are deliberately
         * ignored because they do not read the separate host allocation.
         *
         * @throws std::runtime_error when event ownership or completion is invalid.
         */
        static void waitForPendingHostSourceUseLocked(TensorBase *tensor);

        /// Resolve a backend for the given device.
        /// Uses injected resolver if set, otherwise the global BackendManager.
        IBackend *resolveBackend(DeviceId device) const;

        /**
         * @brief Validate and resolve a mapped signal alias for kernel capture.
         * @return Stable aligned device alias owned by @p region and @p device.
         */
        [[nodiscard]] void *resolveMappedTimelineKernelSignal64(
            const MappedHostTransferRegion &region,
            size_t signal_offset,
            std::uint64_t value,
            DeviceId device,
            const char *operation) const;

        /// Execute HOST_TO_DEVICE transfer
        TransferResult executeHostToDevice(const TransferRequest &req);

        /// Execute DEVICE_TO_HOST transfer
        TransferResult executeDeviceToHost(const TransferRequest &req);

        /// Execute DEVICE_TO_DEVICE_SAME_BACKEND transfer
        TransferResult executeDeviceToDeviceSameBackend(const TransferRequest &req);

        /// Execute HOST_STAGED transfer (generic cross-vendor bounce)
        TransferResult executeHostStaged(const TransferRequest &req);

        /// Log transfer if tracing is enabled
        void traceTransfer(const TransferRequest &req, const TransferResult &result) const;

        /// Wait for a GPU completion event.
        static bool waitForEventWithProxy(IBackend *backend, void *event, int device_id,
                                          const DeviceId &gpu_device);

        BackendResolver resolve_ = nullptr;
    };

    /// Create a MemoryDescriptor from a TensorBase.
    /// This reads all pointer state from the tensor without modifying it.
    MemoryDescriptor makeMemoryDescriptor(const TensorBase *tensor);

} // namespace llaminar2
