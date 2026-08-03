#pragma once

#include <string>

#include "backends/DeviceId.h"
#include "tensors/CoherenceState.h"
#include "transfer/TransferMethod.h"

// Forward declarations
namespace llaminar2
{
    class IBackend;
    class ITensor;
    class TensorBase;
} // namespace llaminar2

namespace llaminar2
{

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
        /// Handles mapped, event wait, device migration, allocation, H2D.
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
        /// Resolve a backend for the given device.
        /// Uses injected resolver if set, otherwise the global BackendManager.
        IBackend *resolveBackend(DeviceId device) const;

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
