/**
 * @file IBackend.h
 * @brief Abstract compute backend interface for GPU operations
 *
 * **Purpose**: Type-safe abstraction over CUDA/ROCm without exposing GPU headers.
 * Prevents header conflicts between cuda_runtime.h and hip_runtime.h.
 *
 * **Phase 3 Objective**: Enable separate compilation units for CUDA (.cu) and ROCm (.cpp)
 * backends while maintaining a unified API.
 *
 * **Design Principles**:
 * - No GPU-specific types in this header (no dim3, float4, cudaError_t, hipError_t)
 * - Pure virtual interface with pointer-based memory operations
 * - Implementations live in backend-specific compilation units
 *
 * @author David Sanftenberg
 */

#pragma once

#include "DeviceType.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace llaminar2
{
    class TransferEngine;
    struct MappedTransferProgressClaim;
    struct MappedTransferProgressCommand;
    struct MappedTransferProgressCompletion;
    struct MappedTransferServiceCursor;
    enum class MappedTransferInterval : std::uint32_t;
    enum class MappedTransferServiceRun : std::uint8_t;
    enum class MappedTransferDirection : std::uint8_t;

    /**
     * @brief One backend-owned device-memory observation at a lifecycle edge.
     *
     * Driver-visible free memory includes every live allocation and every
     * runtime cache.  The graph and default asynchronous-pool fields split out
     * the two caches that CUDA/HIP expose directly.  A false availability flag
     * means that the installed runtime cannot account for that cache; it never
     * means that a zero byte measurement was observed.
     */
    struct DeviceMemoryCacheSnapshot
    {
        size_t driver_free_bytes = 0u; ///< Free bytes reported by the GPU driver.
        size_t graph_used_bytes = 0u; ///< Bytes still owned by graph allocations.
        size_t graph_reserved_bytes = 0u; ///< Graph allocator backing bytes.
        size_t async_pool_used_bytes = 0u; ///< Live default-pool allocations.
        size_t async_pool_reserved_bytes = 0u; ///< Default-pool backing bytes.
        bool graph_accounting_available = false; ///< Graph attributes were queried.
        bool async_pool_accounting_available = false; ///< Pool attributes were queried.
    };

    /**
     * @brief Low-level result of trimming unused CUDA/HIP runtime caches.
     *
     * This type deliberately does not claim that live model allocations were
     * reclaimed.  Backend trim APIs may return only unused graph or async-pool
     * reservations; ownership teardown must precede this call.  Production
     * callers consume it through TransferEngine, which adds the lifecycle
     * intent and minimum-delta certificate.
     */
    struct DeviceMemoryCacheReclamationResult
    {
        bool supported = false; ///< Backend implements scoped cache reclamation.
        bool success = false; ///< Every required query/trim operation succeeded.
        bool graph_trim_invoked = false; ///< Graph cache trim reached the runtime.
        bool async_pool_trim_invoked = false; ///< Default-pool trim reached the runtime.
        DeviceMemoryCacheSnapshot before; ///< Observation before cache trimming.
        DeviceMemoryCacheSnapshot after; ///< Observation after cache trimming.
        std::string diagnostic; ///< Precise failure or capability diagnostic.
    };

    /**
     * @brief Exact canonical allocator ownership for one physical GPU.
     *
     * Driver free-memory is affected by runtime metadata, graph pools, and
     * allocator caching.  This observation instead sums the live allocations
     * made through the backend's canonical allocator, so a model-retirement
     * ticket can prove its exact BOM without guessing fresh-context overhead.
     */
    struct DeviceAllocationAccounting
    {
        bool supported = false; ///< Backend exposes canonical allocation truth.
        size_t active_allocations = 0u; ///< Number of live allocation owners.
        size_t active_bytes = 0u; ///< Exact sum of live allocation byte counts.
        std::string diagnostic; ///< Precise failure or capability diagnostic.
    };

    /**
     * @brief Exact device-address-space reach required by one mapped host region.
     *
     * Device-local registration updates only the named GPU's address space.
     * Backend-portable registration is reserved for a region whose immutable
     * endpoint set contains two or more devices from the same backend family.
     * Keeping this policy typed prevents a one-device activation ticket from
     * needlessly modifying every local GPU page table.
     */
    enum class MappedHostRegistrationScope : std::uint8_t
    {
        DeviceLocal = 0, ///< Only the registration device consumes the pages.
        BackendPortable, ///< Multiple devices in one backend consume the pages.
    };

    /** @return Stable diagnostic spelling for mapped registration reach. */
    [[nodiscard]] constexpr const char *to_string(
        MappedHostRegistrationScope scope) noexcept
    {
        switch (scope)
        {
        case MappedHostRegistrationScope::DeviceLocal:
            return "device_local";
        case MappedHostRegistrationScope::BackendPortable:
            return "backend_portable";
        }
        return "unknown";
    }

    /**
     * @brief Unforgeable authority to retire one exclusive GPU runtime generation.
     *
     * Only TransferEngine can construct this request, after consuming an
     * exclusive model-retirement ticket and retiring the process-local worker
     * context generation. Keeping the constructor private prevents ordinary
     * backend callers from turning a device reset into an ad hoc memory-recovery
     * mechanism.
     */
    class DeviceRuntimeGenerationRetirementRequest final
    {
    public:
        /** @return Backend-local CUDA/HIP ordinal whose generation is retiring. */
        [[nodiscard]] int deviceOrdinal() const noexcept
        {
            return device_ordinal_;
        }

    private:
        friend class TransferEngine;

        /** @brief Construct only at TransferEngine's exclusive ownership edge. */
        explicit DeviceRuntimeGenerationRetirementRequest(
            int device_ordinal) noexcept
            : device_ordinal_(device_ordinal)
        {
        }

        int device_ordinal_ = -1; ///< Exact backend-local physical device.
    };

    /**
     * @brief Certified native-runtime state after an exclusive device reset.
     *
     * The successor generation exists as a host-side cache identity before
     * model execution materializes runtime state for it. Keeping that state
     * explicit prevents a diagnostic memory query from silently rehydrating
     * the runtime generation that model retirement was required to release.
     * CUDA can additionally prove that its primary context is inactive. AMD
     * HIP deliberately keeps a process primary-context object active, so its
     * equivalent proof is a successful native reset followed by no
     * materializing runtime operation.
     */
    enum class DeviceRuntimePostResetState : std::uint8_t
    {
        Unverified = 0, ///< No backend proof of the post-reset native state.
        Quiescent, ///< Reset successor has not been rehydrated for execution.
    };

    /** @return Stable diagnostic spelling for a post-reset runtime state. */
    [[nodiscard]] constexpr const char *to_string(
        DeviceRuntimePostResetState state) noexcept
    {
        switch (state)
        {
        case DeviceRuntimePostResetState::Unverified:
            return "unverified";
        case DeviceRuntimePostResetState::Quiescent:
            return "quiescent";
        }
        return "unknown";
    }

    /**
     * @brief Backend proof for one completed CUDA/HIP runtime reset generation.
     *
     * A successful result means the backend proved that no tracked device
     * allocation or host registration remained, invalidated its own cached
     * runtime handles, reset the named device context, and published a new
     * monotonically increasing successor generation. The successor must remain
     * quiescent: querying driver free memory after reset through the runtime
     * can rehydrate it and defeat reclamation. Driver free memory is
     * therefore observed only before reset; the next legitimate model
     * admission observes capacity while activating the successor generation.
     */
    struct DeviceRuntimeGenerationRetirementResult
    {
        bool supported = false; ///< Backend implements explicit generation reset.
        bool success = false; ///< Every precondition, reset, and publication passed.
        bool reset_invoked = false; ///< Native cuda/hip device reset was called.
        std::uint64_t retired_generation = 0u; ///< Generation invalidated by reset.
        std::uint64_t successor_generation = 0u; ///< Published quiescent generation.
        DeviceRuntimePostResetState post_reset_state =
            DeviceRuntimePostResetState::Unverified; ///< Native successor state.
        size_t driver_free_bytes_before = 0u; ///< Driver free bytes before reset.
        size_t tracked_device_allocations = 0u; ///< Live allocations at preflight.
        size_t tracked_device_allocation_bytes = 0u; ///< Live bytes at preflight.
        size_t tracked_host_registrations = 0u; ///< Live registrations at preflight.
        std::string diagnostic; ///< Precise failure or completion diagnostic.
    };

    namespace sampling_math
    {
        struct DeviceGenerationDepthPolicy;
        enum class DeviceGenerationLeadingRowDisposition : int32_t;
    }


    /**
     * @class IBackend
     * @brief Abstract interface for compute backend operations
     *
     * **Implementations**:
     * - `CUDABackend` (backends/cuda/CUDABackend.cu)
     * - `ROCmBackend` (backends/rocm/ROCmBackend.cpp)
     * - `CPUBackend` (backends/CPUBackend.cpp) - for completeness
     *
     * **Usage**:
     * ```cpp
     * IBackend* backend = nullptr;
     * #ifdef HAVE_CUDA
     *     backend = new CUDABackend();
     * #elif HAVE_ROCM
     *     backend = new ROCmBackend();
     * #endif
     *
     * if (backend) {
     *     backend->deviceToHost(host_ptr, device_ptr, bytes, device_id, stream);
     *     backend->synchronize(device_id);
     * }
     * ```
     */
    class IBackend
    {
    public:
        virtual ~IBackend() = default;

        // ====================================================================
        // Memory Transfer Operations
        // ====================================================================

        /**
         * @brief Copy data from device to host
         *
         * @param dst Host destination pointer (must be pre-allocated)
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * This is a low-level compatibility boundary for raw buffers. GPU
         * implementations enqueue the copy, record one exact completion event,
         * and wait only that event before returning host-owned bytes. Tensor
         * callers must use TransferEngine so coherence and source lifetimes are
         * published instead of hidden behind this blocking boundary.
         *
         * @param stream Exact producer stream for GPU backends. GPU
         *               implementations reject nullptr. CPU callers pass
         *               nullptr explicitly because CPU execution is synchronous.
         */
        virtual bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id, void *stream) = 0;

        /**
         * @brief Fast D2H copy — skips pointer validation for hot paths
         *
         * Caller guarantees:
         * - src is a valid device pointer on device_id
         * - GPU work writing to src is ordered before the exact supplied stream
         * - dst is a valid host pointer with sufficient space
         *
         * Default implementation delegates to deviceToHost().
         * ROCm override skips hipPointerGetAttributes() + HipDeviceSaveRestore (~30-60µs savings).
         */
        virtual bool deviceToHostFast(void *dst, const void *src, size_t bytes, int device_id, void *stream)
        {
            return deviceToHost(dst, src, bytes, device_id, stream);
        }

        /**
         * @brief Copy data from host to device
         *
         * @param dst Device destination pointer (must be pre-allocated)
         * @param src Host source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * This is a low-level compatibility boundary for raw buffers. GPU
         * implementations enqueue the copy and wait one exact completion event;
         * they never synchronize the complete stream. Tensor callers must use
         * TransferEngine, which retains the completion event asynchronously.
         *
         * @param stream Exact consumer stream for GPU backends. GPU
         *               implementations reject nullptr. CPU callers pass
         *               nullptr explicitly because CPU execution is synchronous.
         */
        virtual bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream) = 0;

        /**
         * @brief Copy data between two device pointers on the same device
         *
         * @param dst Device destination pointer
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)
         * - ROCm: hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice)
         * - CPU: memcpy(dst, src, bytes)
         *
         * Used for device-to-device tensor transfers where both src and dst
         * are in the same GPU's VRAM (standard device memory pointers).
         *
         * @param stream Exact operation stream for GPU backends. GPU
         *               implementations reject nullptr.
         */
        virtual bool deviceToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream)
        {
            // Default implementation: not supported
            (void)dst;
            (void)src;
            (void)bytes;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Synchronize all operations on a device
         *
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaDeviceSynchronize()
         * - ROCm: hipDeviceSynchronize()
         * - CPU: no-op (always synchronous)
         */
        virtual bool synchronize(int device_id) = 0;

        /**
         * @brief Synchronize the default stream on a device
         *
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaStreamSynchronize(nullptr)
         * - ROCm: hipStreamSynchronize(nullptr)
         * - CPU: no-op (always synchronous)
         *
         * **Performance Note**: This is lighter than synchronize() because it only
         * waits for the default stream, not all streams on the device.
         */
        virtual bool streamSynchronize(int device_id) = 0;

        // ====================================================================
        // Event Operations (Fine-grained Synchronization)
        // ====================================================================

        /**
         * @brief Create an event for fine-grained synchronization
         *
         * @param device_id GPU device ID (0-based)
         * @return Opaque event handle (nullptr on failure)
         *
         * **Semantics**:
         * - CUDA: cudaEventCreate()
         * - ROCm: hipEventCreate()
         * - CPU: returns dummy non-null pointer
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         * **Lifetime**: Caller owns the event and must call destroyEvent()
         */
        virtual void *createEvent(int device_id) = 0;

        /**
         * @brief Create an event suitable for elapsed-time measurement.
         *
         * Normal synchronization events may disable timing to avoid pipeline
         * overhead in the hot path.  Profiling code that needs device elapsed
         * time must request timing-capable events explicitly through this API.
         *
         * @param device_id GPU device ID (0-based)
         * @return Opaque event handle (nullptr on failure)
         */
        virtual void *createTimingEvent(int device_id)
        {
            return createEvent(device_id);
        }

        /**
         * @brief Destroy an event created by createEvent()
         *
         * @param event Opaque event handle (may be nullptr)
         * @param device_id GPU device ID (0-based)
         *
         * **Semantics**:
         * - CUDA: cudaEventDestroy()
         * - ROCm: hipEventDestroy()
         * - CPU: no-op
         */
        virtual void destroyEvent(void *event, int device_id) = 0;

        /**
         * @brief Record an event on the specified stream
         *
         * Marks the current point in the given stream. All operations
         * submitted before this call on that stream will complete before
         * the event is signaled.
         *
         * @param event Opaque event handle from createEvent()
         * @param device_id GPU device ID (0-based)
         * @param stream Exact operation stream for GPU backends. GPU
         *               implementations reject nullptr.
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaEventRecord(event, stream)
         * - ROCm: hipEventRecord(event, stream)
         * - CPU: no-op (returns true)
         */
        virtual bool recordEvent(void *event, int device_id, void *stream) = 0;

        /**
         * @brief Wait for an event to complete
         *
         * Blocks the host until all operations recorded before the event
         * have completed on the device.
         *
         * @param event Opaque event handle from createEvent()
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaEventSynchronize(event)
         * - ROCm: hipEventSynchronize(event)
         * - CPU: no-op (returns true)
         *
         * **Performance Note**: This is lighter than synchronize() because it only
         * waits for a specific point in the stream, not all device operations.
         */
        virtual bool waitForEvent(void *event, int device_id) = 0;

        /**
         * @brief Query an event without blocking the calling host thread.
         *
         * This is the observation primitive for bounded lifecycle protocols
         * that must wait for an exact producer-stream point without calling a
         * stream- or device-wide synchronization API.  A successful query may
         * report either ready or not-ready; backend/runtime errors are reported
         * by returning false.  Implementations must never turn a not-ready
         * result into a blocking event wait.
         *
         * @param event Opaque event handle returned by createEvent().
         * @param device_id Device ordinal that owns @p event.
         * @param ready Non-null destination; set true only after the recorded
         *              stream work has completed.
         * @return true when the query itself succeeded, including not-ready.
         */
        virtual bool queryEvent(void *event, int device_id, bool *ready)
        {
            (void)event;
            (void)device_id;
            if (ready)
                *ready = false;
            return false;
        }

        /**
         * @brief Measure elapsed milliseconds between two recorded timing events.
         *
         * The caller must ensure both events came from @ref createTimingEvent
         * and have completed before relying on the value.  Implementations
         * return false when elapsed timing is unsupported.
         *
         * @param start_event Event recorded before the measured work
         * @param stop_event Event recorded after the measured work
         * @param device_id GPU device ID (0-based)
         * @param out_ms Destination for elapsed milliseconds
         * @return true on success, false on unsupported/error
         */
        virtual bool eventElapsedTimeMs(
            void *start_event,
            void *stop_event,
            int device_id,
            float *out_ms)
        {
            (void)start_event;
            (void)stop_event;
            (void)device_id;
            (void)out_ms;
            return false;
        }

        /**
         * @brief Set active device for subsequent operations
         *
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaSetDevice(device_id)
         * - ROCm: hipSetDevice(device_id)
         * - CPU: no-op (single device)
         */
        virtual bool setDevice(int device_id) = 0;

        // ====================================================================
        // Memory Allocation Operations
        // ====================================================================

        /**
         * @brief Allocate device memory
         *
         * @param bytes Number of bytes to allocate
         * @param device_id GPU device ID (0-based)
         * @return Pointer to allocated memory (nullptr on failure)
         *
         * **Semantics**:
         * - CUDA: cudaMalloc(&ptr, bytes)
         * - ROCm: hipMalloc(&ptr, bytes)
         * - CPU: malloc(bytes)
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         */
        virtual void *allocate(size_t bytes, int device_id) = 0;

        /**
         * @brief Free device memory
         *
         * @param ptr Device pointer to free (may be nullptr)
         * @param device_id GPU device ID (0-based)
         *
         * **Semantics**:
         * - CUDA: cudaFree(ptr)
         * - ROCm: hipFree(ptr)
         * - CPU: free(ptr)
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         */
        virtual void free(void *ptr, int device_id) = 0;

        /**
         * @brief Set device memory to a byte value
         *
         * @param ptr Device pointer to fill
         * @param value Byte value to set (0-255)
         * @param bytes Number of bytes to set
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemset(ptr, value, bytes)
         * - ROCm: hipMemset(ptr, value, bytes)
         * - CPU: memset(ptr, value, bytes)
         *
         * **Common Use Cases**:
         * - Zero-initialize output buffers before kernel execution
         * - Set corruption patterns (0xCC) for data integrity verification
         * - Clear buffers between operations in P2P testing
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         *
         * @param stream Exact operation stream for GPU backends. GPU
         *               implementations reject nullptr.
         */
        virtual bool memset(void *ptr, int value, size_t bytes, int device_id, void *stream) = 0;

        /**
         * @brief Materialize the next fixed-width prefill bucket from admitted device state.
         *
         * The primitive derives the relative source row from
         * `*cached_tokens_device - request_position_ids_device[0]`. The canonical
         * KV count is therefore the only progression cursor. It writes one stable
         * token bucket, one stable absolute-position bucket, the current real row
         * count, and the physical row stride. Inactive rows receive @p pad_token_id
         * and deterministic continuation positions.
         *
         * GPU implementations must enqueue exactly one graph-capturable,
         * allocation-free, transfer-free kernel on the explicit non-null stream.
         * Invalid live geometry is fatal device state and must trap rather than
         * clamp, retry, or select a host-authored path.
         *
         * @param request_token_ids_device Complete admitted INT32 token bank.
         * @param request_position_ids_device Complete admitted INT32 position bank.
         * @param request_total_rows_device Device INT32 logical request length.
         * @param cached_tokens_device Canonical device INT32 main-KV count.
         * @param request_row_capacity Number of rows available in the admitted bank.
         * @param bucket_seq_len Physical rows materialized on every graph replay.
         * @param pad_token_id Token id written to inactive bucket rows.
         * @param device_id GPU ordinal owning every pointer.
         * @param stream Exact graph execution stream.
         * @param out_token_ids_device Stable INT32 token bucket.
         * @param out_position_ids_device Stable INT32 absolute-position bucket.
         * @param out_real_rows_device Device INT32 logical row count for this bucket.
         * @param out_row_stride_device Device INT32 physical bucket stride.
         * @return true when the kernel was enqueued successfully.
         */
        virtual bool enqueuePreparePrefillChunkView(
            const void *request_token_ids_device,
            const void *request_position_ids_device,
            const void *request_total_rows_device,
            const void *cached_tokens_device,
            int request_row_capacity,
            int bucket_seq_len,
            int pad_token_id,
            int device_id,
            void *stream,
            void *out_token_ids_device,
            void *out_position_ids_device,
            void *out_real_rows_device,
            void *out_row_stride_device)
        {
            (void)request_token_ids_device;
            (void)request_position_ids_device;
            (void)request_total_rows_device;
            (void)cached_tokens_device;
            (void)request_row_capacity;
            (void)bucket_seq_len;
            (void)pad_token_id;
            (void)device_id;
            (void)stream;
            (void)out_token_ids_device;
            (void)out_position_ids_device;
            (void)out_real_rows_device;
            (void)out_row_stride_device;
            return false;
        }

        // ====================================================================
        // Zero-Copy Mapped Memory Operations (GPU writes directly to host)
        // ====================================================================

        /**
         * @brief Allocate zero-copy mapped memory (host memory accessible by GPU)
         *
         * Allocates pinned host memory that can be written to directly by GPU kernels.
         * The GPU writes via PCIe, so no explicit D2H memcpy is needed - but writes
         * are slower than to device memory (PCIe bandwidth limited).
         *
         * @param bytes Number of bytes to allocate
         * @param device_id GPU device ID that will access this memory
         * @param[out] device_ptr Receives the device-visible pointer for the GPU
         * @return Host pointer (nullptr on failure). device_ptr set to GPU-visible address.
         *
         * **Use Case**: Output tensors that need to be read by host (verification, snapshots).
         * Instead of: kernel writes to device mem -> hipMemcpy D2H
         * Do: kernel writes directly to mapped host mem -> hipDeviceSynchronize -> read host
         *
         * **Performance Trade-off**:
         * - Pro: Eliminates D2H memcpy entirely
         * - Pro: Can read partial data while kernel is running (with care)
         * - Con: GPU writes to host are PCIe-limited (~15 GB/s vs ~1 TB/s HBM)
         *
         * **Semantics**:
         * - CUDA: cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped)
         *         cudaHostGetDevicePointer(&dev_ptr, ptr, 0)
         * - ROCm: hipHostMalloc(&ptr, bytes, hipHostMallocMapped)
         *         hipHostGetDevicePointer(&dev_ptr, ptr, 0)
         * - CPU: malloc(bytes), device_ptr = nullptr (no GPU)
         *
         * **Thread Safety**: Caller must ensure device is set before calling
         * **Lifetime**: Caller owns memory, must call freeMapped()
         */
        virtual void *allocateMapped(size_t bytes, int device_id, void **device_ptr) = 0;

        /**
         * @brief Free zero-copy mapped memory allocated by allocateMapped()
         *
         * @param host_ptr Host pointer returned by allocateMapped() (may be nullptr)
         * @param device_id GPU device ID used for allocation
         *
         * **Semantics**:
         * - CUDA: cudaFreeHost(host_ptr)
         * - ROCm: hipHostFree(host_ptr)
         * - CPU: free(host_ptr)
         */
        virtual void freeMapped(void *host_ptr, int device_id) = 0;

        // ====================================================================
        // Device Query Operations
        // ====================================================================

        /**
         * @brief Get number of available devices
         *
         * @return Device count (0 if backend unavailable)
         *
         * **Semantics**:
         * - CUDA: cudaGetDeviceCount()
         * - ROCm: hipGetDeviceCount()
         * - CPU: Always returns 1
         */
        virtual int deviceCount() const = 0;

        /**
         * @brief Get backend name (for logging/debugging)
         *
         * @return Backend identifier ("CUDA", "ROCm", "CPU")
         */
        virtual std::string backendName() const = 0;

        /**
         * @brief Get device name string
         *
         * @param device_id GPU device ID (0-based)
         * @return Device name (e.g., "NVIDIA A100", "AMD MI250X")
         */
        virtual std::string deviceName(int device_id) const = 0;

        /**
         * @brief Get total device memory in bytes
         *
         * @param device_id GPU device ID (0-based)
         * @return Total memory in bytes (0 on error)
         */
        virtual size_t deviceMemoryTotal(int device_id) const = 0;

        /**
         * @brief Get free device memory in bytes
         *
         * @param device_id GPU device ID (0-based)
         * @return Free memory in bytes (0 on error)
         */
        virtual size_t deviceMemoryFree(int device_id) const = 0;

        /**
         * @brief Observe exact allocations owned by the canonical backend.
         *
         * GPU implementations must serialize this observation against
         * allocation/free and native runtime reset. The default is unsupported
         * so an unaccounted backend cannot certify exclusive model retirement.
         */
        [[nodiscard]] virtual DeviceAllocationAccounting
        deviceAllocationAccounting(int device_id) const
        {
            (void)device_id;
            DeviceAllocationAccounting accounting;
            accounting.diagnostic =
                "backend does not expose canonical device-allocation accounting";
            return accounting;
        }

        /**
         * @brief Return unused backend runtime caches to the device allocator.
         *
         * This infrastructure hook is intentionally narrower than a device
         * reset: it must preserve live allocations, contexts, streams, events,
         * graphs, and library handles owned by other models.  Callers must
         * first retire the exact execution topology whose caches they expect
         * to release.  Production code calls TransferEngine rather than this
         * method directly.
         *
         * CPU and test backends inherit an unsupported result unless they
         * explicitly model this lifecycle operation.
         *
         * @param device_id Backend-local GPU ordinal.
         * @return Before/after accounting plus typed success state.
         */
        virtual DeviceMemoryCacheReclamationResult
        trimUnusedDeviceMemoryCaches(int device_id)
        {
            (void)device_id;
            DeviceMemoryCacheReclamationResult result;
            result.diagnostic =
                "backend does not support scoped device-memory cache reclamation";
            return result;
        }

        /**
         * @brief Retire one exclusive CUDA/HIP runtime generation.
         *
         * This is an infrastructure-only primitive. Production code must call
         * TransferEngine's two-phase exclusive model-retirement API, which is
         * the only authority able to construct @p request. Implementations must
         * reject the operation while any backend-tracked allocation or host
         * registration remains, clear every backend-owned cached runtime handle,
         * publish a quiescent successor generation only after the native reset
         * succeeds, and certify without reactivating that native context.
         *
         * CPU and test backends inherit an unsupported result unless they model
         * this lifecycle explicitly.
         *
         * @param request Unforgeable exact-device retirement authority.
         * @return Typed reset and generation-publication evidence.
         */
        virtual DeviceRuntimeGenerationRetirementResult
        retireExclusiveDeviceRuntimeGeneration(
            const DeviceRuntimeGenerationRetirementRequest &request)
        {
            (void)request;
            DeviceRuntimeGenerationRetirementResult result;
            result.diagnostic =
                "backend does not support exclusive runtime-generation retirement";
            return result;
        }

        /**
         * @brief Return the current runtime generation for one physical device.
         *
         * Persistent host-side caches that mirror device modules or handles use
         * this identity in their cache keys. Zero means the backend does not
         * expose reset generations and must never be treated as a live GPU
         * generation.
         *
         * @param device_id Backend-local device ordinal.
         * @return Non-zero active generation for supporting GPU backends.
         */
        [[nodiscard]] virtual std::uint64_t
        deviceRuntimeGeneration(int device_id) const
        {
            (void)device_id;
            return 0u;
        }

        // ====================================================================
        // Capability Queries
        // ====================================================================

        /**
         * @brief Check if backend supports BF16 compute
         *
         * @param device_id GPU device ID (0-based)
         * @return true if BF16 supported (e.g., CUDA compute capability ≥ 8.0)
         */
        virtual bool supportsBF16(int device_id) const = 0;

        /**
         * @brief Check if backend supports FP16 compute
         *
         * @param device_id GPU device ID (0-based)
         * @return true if FP16 supported (e.g., CUDA compute capability ≥ 5.3)
         */
        virtual bool supportsFP16(int device_id) const = 0;

        /**
         * @brief Check if backend supports INT8 compute
         *
         * @param device_id GPU device ID (0-based)
         * @return true if INT8 supported (e.g., CUDA compute capability ≥ 6.1)
         */
        virtual bool supportsINT8(int device_id) const = 0;

        // ====================================================================
        // Host Memory Pinning (for async DMA)
        // ====================================================================

        /**
         * @brief Pin host memory for zero-copy DMA transfers
         *
         * Pinned (page-locked) memory enables true async DMA transfers and
         * eliminates internal staging copies in hipMemcpy/cudaMemcpy.
         *
         * @param ptr Host pointer to pin
         * @param bytes Size of the region in bytes
         * @param device_id Exact backend-local device whose runtime context
         *                  owns the registration lifecycle
         * @return true on success, false on error
         *
         * Call unpinHostMemory() with the same device when done. No-op on CPU
         * backend. Implementations must bind the named device explicitly;
         * ambient thread-local device state is not an ownership contract.
         */
        virtual bool pinHostMemory(void *ptr, size_t bytes, int device_id)
        {
            (void)ptr;
            (void)bytes;
            (void)device_id;
            return true;
        }

        /**
         * @brief Retire a host-memory registration in its owning device context.
         * @param ptr Exact address passed to @ref pinHostMemory.
         * @param device_id Exact backend-local registration owner.
         * @return true only when the registration is no longer live.
         */
        virtual bool unpinHostMemory(void *ptr, int device_id)
        {
            (void)ptr;
            (void)device_id;
            return true;
        }

        /**
         * @brief Register caller-owned pages with exact typed device reach.
         *
         * This is distinct from ordinary DMA pinning: the registration must be
         * device-visible alias through @ref externalMappedHostDevicePointer.
         * Device-local scope must not update unrelated device page tables;
         * backend-portable scope is valid only when the caller has declared
         * multiple consumers in this backend family. The call is setup-only
         * and must not allocate or copy payload bytes.
         *
         * @param ptr Stable page-aligned or runtime-acceptable host address.
         * @param bytes Positive immutable region size.
         * @param registration_device_id One valid local device used to establish
         *        the backend registration context.
         * @param scope Exact address-space reach required by the declared endpoints.
         * @return True only when the complete region was registered with that reach.
         */
        virtual bool registerExternalMappedHostMemory(
            void *ptr,
            size_t bytes,
            int registration_device_id,
            MappedHostRegistrationScope scope)
        {
            (void)ptr;
            (void)bytes;
            (void)registration_device_id;
            (void)scope;
            return false;
        }

        /**
         * @brief Resolve one exact device alias for a mapped external host region.
         * @param host_ptr Address previously registered by this backend.
         * @param device_id Local device whose address space will consume the alias.
         * @param[out] device_ptr Non-null device-visible address on success.
         */
        virtual bool externalMappedHostDevicePointer(
            void *host_ptr,
            int device_id,
            void **device_ptr)
        {
            (void)host_ptr;
            (void)device_id;
            if (device_ptr)
                *device_ptr = nullptr;
            return false;
        }

        /**
         * @brief Unregister mapped external pages after every device stream drained.
         * @param ptr Exact registered host address.
         * @param registration_device_id Context ordinal used at registration.
         */
        virtual bool unregisterExternalMappedHostMemory(
            void *ptr,
            int registration_device_id)
        {
            (void)ptr;
            (void)registration_device_id;
            return false;
        }

        /**
         * @brief GPU-side argmax over FP32 data
         *
         * Finds the index and value of the maximum element entirely on the GPU,
         * avoiding a full D2H transfer of the logits tensor.
         * Used for greedy decode sampling.
         *
         * @param data_device Device pointer to FP32 data
         * @param n Number of elements
         * @param device_id Device where data resides
         * @param out_value Receives the maximum value
         * @param out_index Receives the index of the maximum element
         * @param stream Optional device stream to enqueue work on
         * @param partial_vals Device scratch [partial_capacity] for the
         *        two-pass multi-block reduction (per-block partial max values).
         *        Production GPU backends require caller-owned workspace/arena
         *        scratch and fail loud when it is missing or undersized.
         * @param partial_idxs Device scratch [partial_capacity] for the
         *        per-block partial max indices (paired with @p partial_vals).
         * @param partial_capacity Number of entries in the partial scratch buffers.
         * @return true if executed on device, false if not supported (caller should fall back)
         */
        virtual bool argmaxF32(const void *data_device, int n, int device_id,
                               float *out_value, int *out_index, void *stream,
                               void *partial_vals = nullptr, void *partial_idxs = nullptr,
                               int partial_capacity = 0)
        {
            (void)data_device;
            (void)n;
            (void)device_id;
            (void)out_value;
            (void)out_index;
            (void)stream;
            (void)partial_vals;
            (void)partial_idxs;
            (void)partial_capacity;
            return false; // Not supported by default
        }

        /**
         * @brief GPU-side greedy argmax for several contiguous FP32 rows.
         *
         * @param data_device Device pointer to row-major FP32 data.
         * @param rows Number of rows to sample.
         * @param cols Number of columns per row.
         * @param device_id Device where data resides.
         * @param out_values Host buffer [rows] for max values.
         * @param out_indices Host buffer [rows] for row-local argmax indices.
         * @param stream Optional device stream to enqueue work on.
         * @param partial_vals Device scratch shared across rows. Backends that
         *        implement a fused batched kernel should partition this scratch
         *        internally. The default implementation calls argmaxF32() once
         *        per row, preserving existing backend behavior.
         * @param partial_idxs Device scratch paired with @p partial_vals.
         * @param partial_capacity Number of entries in each scratch buffer.
         * @return true if every row was sampled on device.
         */
        virtual bool argmaxF32BatchedRows(const void *data_device, int rows, int cols, int device_id,
                                          float *out_values, int *out_indices, void *stream,
                                          void *partial_vals = nullptr, void *partial_idxs = nullptr,
                                          int partial_capacity = 0)
        {
            if (!data_device || rows <= 0 || cols <= 0 || !out_values || !out_indices)
                return false;

            const auto *base = static_cast<const float *>(data_device);
            for (int row = 0; row < rows; ++row)
            {
                if (!argmaxF32(base + static_cast<size_t>(row) * static_cast<size_t>(cols),
                               cols,
                               device_id,
                               out_values + row,
                               out_indices + row,
                               stream,
                               partial_vals,
                               partial_idxs,
                               partial_capacity))
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Enqueue GPU-side greedy argmax for several contiguous FP32 rows.
         *
         * Unlike argmaxF32BatchedRows(), this vLLM-style primitive leaves the
         * row-local argmax values and indices on device. It performs no
         * allocation, host copy, or synchronization, and is suitable for graph
         * capture plus downstream device-side speculative verification summary.
         *
         * @param out_values_device Device buffer [rows] for max values.
         * @param out_indices_device Device buffer [rows] for row-local token ids.
         * @param output_stride Element stride between adjacent output rows.
         */
        virtual bool enqueueArgmaxF32BatchedRowsDevice(
            const void *data_device,
            int rows,
            int cols,
            int device_id,
            void *stream,
            void *out_values_device,
            void *out_indices_device,
            void *partial_vals = nullptr,
            void *partial_idxs = nullptr,
            int partial_capacity = 0,
            int output_stride = 1)
        {
            (void)data_device;
            (void)rows;
            (void)cols;
            (void)device_id;
            (void)stream;
            (void)out_values_device;
            (void)out_indices_device;
            (void)partial_vals;
            (void)partial_idxs;
            (void)partial_capacity;
            (void)output_stride;
            return false;
        }

        /**
         * @brief Enqueue deterministic argmax and publish the next MTP sidecar inputs.
         *
         * This is the producer-owned form used by a captured MTP proposal graph.
         * The final reduction writes the ordinary row-local value/token outputs,
         * mirrors each winning token into the chained-sidecar condition mailbox,
         * and advances that row's resident position in the same kernel. Keeping
         * these stores in the argmax finalizer makes the proposal-to-sidecar edge
         * indivisible and removes a separate copy/position kernel from every draft.
         *
         * Implementations must preserve the exact reduction and lowest-token tie
         * semantics of enqueueArgmaxF32BatchedRowsDevice(). No allocation,
         * transfer, host observation, event operation, or synchronization is
         * permitted. All pointers are device-resident and @p stream is mandatory.
         */
        virtual bool enqueueArgmaxF32BatchedRowsAndPublishMTPChainDevice(
            const void *data_device,
            int rows,
            int cols,
            int device_id,
            void *stream,
            void *out_values_device,
            void *out_indices_device,
            void *chain_condition_tokens_device,
            void *chain_position_ids_device,
            int chain_position_increment,
            void *partial_vals,
            void *partial_idxs,
            int partial_capacity,
            int output_stride = 1)
        {
            (void)data_device;
            (void)rows;
            (void)cols;
            (void)device_id;
            (void)stream;
            (void)out_values_device;
            (void)out_indices_device;
            (void)chain_condition_tokens_device;
            (void)chain_position_ids_device;
            (void)chain_position_increment;
            (void)partial_vals;
            (void)partial_idxs;
            (void)partial_capacity;
            (void)output_stride;
            return false;
        }

        /**
         * @brief Retain one transaction-zero MTP sidecar boundary on device.
         *
         * This diagnostic-only primitive hashes an exact array of 32-bit words
         * into the arena-owned first-transaction record.  The device generation
         * controller gates the write, so later iterations of a captured WHILE
         * graph cannot overwrite the first proposal sequence.  A null data
         * pointer with zero words records an explicitly absent model boundary.
         *
         * Implementations must enqueue on @p stream without allocation,
         * transfer, host observation, event operations, or synchronization.
         * Production graphs never call this method unless first-transaction
         * diagnostics were enabled before graph construction.
         */
        virtual bool enqueueRetainMTPFirstTransactionDraftBoundaryDevice(
            const void *data_words_device,
            int word_count,
            int boundary,
            int draft_slot,
            const void *condition_token_device,
            const void *position_id_device,
            const void *generation_control_device,
            int generation_control_stride,
            void *diagnostic_record_device,
            int device_id,
            void *stream)
        {
            (void)data_words_device;
            (void)word_count;
            (void)boundary;
            (void)draft_slot;
            (void)condition_token_device;
            (void)position_id_device;
            (void)generation_control_device;
            (void)generation_control_stride;
            (void)diagnostic_record_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Publish grouped-greedy penalty controls with a device kernel.
         *
         * The policy values are launch arguments, while @p controls_device is a
         * persistent arena address captured by the verifier graph.  Backends
         * must enqueue one non-blocking kernel on the exact non-null stream;
         * host-to-device copies and synchronization are forbidden.
         */
        virtual bool enqueueConfigureMTPGreedyPenaltyPolicyDevice(
            void *controls_device,
            float presence_penalty,
            float frequency_penalty,
            bool first_token_already_in_history,
            int device_id,
            void *stream)
        {
            (void)controls_device;
            (void)presence_penalty;
            (void)frequency_penalty;
            (void)first_token_already_in_history;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Enqueue decode-equivalent grouped argmax with device history.
         *
         * Every verifier row is scored against the persistent generated-token
         * histogram plus only the preceding tokens in that row's speculative
         * branch.  Implementations must preserve serial float operation order
         * and deterministic lowest-token tie breaking.
         *
         * @param active_rows_device Resident INT32 logical verifier width. This
         *        pointer is mandatory for grouped GPU verification: both argmax
         *        passes must ignore the inactive suffix of a larger captured
         *        physical bucket without consulting a host scalar.
         */
        virtual bool enqueueArgmaxF32BatchedRowsWithMTPPenaltiesDevice(
            const void *data_device,
            int rows,
            int cols,
            const void *verifier_input_tokens_device,
            const void *generated_token_counts_device,
            const void *penalty_policy_device,
            const void *active_rows_device,
            int device_id,
            void *stream,
            void *out_values_device,
            void *out_indices_device,
            void *partial_vals,
            void *partial_idxs,
            int partial_capacity,
            int output_stride = 1)
        {
            (void)data_device;
            (void)rows;
            (void)cols;
            (void)verifier_input_tokens_device;
            (void)generated_token_counts_device;
            (void)penalty_policy_device;
            (void)active_rows_device;
            (void)device_id;
            (void)stream;
            (void)out_values_device;
            (void)out_indices_device;
            (void)partial_vals;
            (void)partial_idxs;
            (void)partial_capacity;
            (void)output_stride;
            return false;
        }

        /**
         * @brief Apply decode-equivalent MTP history penalties in-place.
         *
         * The durable generated-token histogram is device-owned request state.
         * When @p verifier_input_tokens_device is non-null, row `r` additionally
         * observes the speculative branch tokens `[prefix_begin, r]`, matching
         * the history visible to serial decode at that verifier row.  A null
         * verifier-input pointer applies only durable history and is used for
         * the first target token of the next transaction.
         *
         * Implementations must enqueue one graph-capturable operation on the
         * exact non-null producer stream.  Allocations, copies, atomics, and
         * synchronization are forbidden.
         * @param active_rows_device Optional resident INT32 logical row count.
         *        The captured @p rows remains physical capacity; work at or
         *        beyond the resident count must be skipped before reading logits
         *        or speculative-prefix tokens.
         */
        virtual bool enqueueApplyMTPPenaltiesToF32RowsDevice(
            void *data_device,
            int rows,
            int cols,
            int row_stride,
            const void *verifier_input_tokens_device,
            const void *generated_token_counts_device,
            const void *penalty_policy_device,
            int device_id,
            void *stream,
            const void *active_rows_device = nullptr)
        {
            (void)data_device;
            (void)rows;
            (void)cols;
            (void)row_stride;
            (void)verifier_input_tokens_device;
            (void)generated_token_counts_device;
            (void)penalty_policy_device;
            (void)device_id;
            (void)stream;
            (void)active_rows_device;
            return false;
        }

        /**
         * @brief Apply one MTP proposal row's complete branch history in-place.
         *
         * The proposal row observes the durable generated-token histogram, the
         * first condition token when it is not already durable, and exactly
         * @p prior_draft_count preceding device-resident draft slots.  This is
         * the device-owned equivalent of cloning the serial sampler and
         * recording the current speculative branch before scoring its next
         * proposal.
         *
         * Implementations must preserve the serial presence-then-frequency
         * arithmetic order and enqueue on the exact non-null producer stream.
         * Allocations, copies, atomics, and synchronization are forbidden.
         */
        virtual bool enqueueApplyMTPBranchPenaltiesToF32RowDevice(
            void *data_device,
            int cols,
            const void *first_condition_token_device,
            const void *prior_draft_tokens_device,
            int prior_draft_count,
            const void *generated_token_counts_device,
            const void *penalty_policy_device,
            int device_id,
            void *stream)
        {
            (void)data_device;
            (void)cols;
            (void)first_condition_token_device;
            (void)prior_draft_tokens_device;
            (void)prior_draft_count;
            (void)generated_token_counts_device;
            (void)penalty_policy_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Commit newly emitted compact outcome tokens to device history.
         *
         * Accepted-state publication is the sole owner of this transition. It
         * advances the mirrored histogram in compact-output order, then derives
         * `first_token_already_in_history` from the final accepted-state count
         * and stopped flag on the same stream.  The mutable policy pointer is a
         * persistent device binding; no host-authored transaction bit is
         * accepted by this interface.
         *
         * Implementations must be graph capturable and use no atomics,
         * allocation, transfer, or synchronization.
         */
        virtual bool enqueueCommitMTPGreedyPenaltyHistoryDevice(
            const void *output_tokens_device,
            const void *output_meta_device,
            void *penalty_policy_device,
            const void *accepted_state_counts_device,
            const void *stopped_flags_device,
            int output_token_capacity,
            int vocab_size,
            void *generated_token_counts_device,
            int device_id,
            void *stream)
        {
            (void)output_tokens_device;
            (void)output_meta_device;
            (void)penalty_policy_device;
            (void)accepted_state_counts_device;
            (void)stopped_flags_device;
            (void)output_token_capacity;
            (void)vocab_size;
            (void)generated_token_counts_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief GPU-side top-k selection over FP32 data
         *
         * Finds the k largest elements (value and index) entirely on the GPU.
         * Results are in descending order of value (highest first).
         * Used for top-k/top-p sampling to avoid a full D2H transfer.
         *
         * @param data_device Device pointer to FP32 data
         * @param n Number of elements
         * @param k Number of top elements to select (1..256)
         * @param device_id Device where data resides
         * @param out_values Host buffer for k float values (descending order)
         * @param out_indices Host buffer for k int indices
         * @return true if executed on device, false if not supported
         */
        virtual bool topKF32(const void *data_device, int n, int k, int device_id,
                             float *out_values, int *out_indices, void *stream)
        {
            (void)data_device;
            (void)n;
            (void)k;
            (void)device_id;
            (void)out_values;
            (void)out_indices;
            (void)stream;
            return false; // Not supported by default
        }

        /**
         * @brief GPU-side top-k/top-p/temperature sampling over FP32 logits.
         *
         * This synchronous convenience wrapper only copies the selected token
         * back to the host. It must not materialize full logits on the CPU.
         *
         * @param data_device Device pointer to FP32 logits [n]
         * @param n Vocabulary size
         * @param top_k Top-k candidate limit (1..256, clamped by backend)
         * @param top_p Nucleus probability threshold (<=0 or >=1 disables)
         * @param temperature Sampling temperature (<=0 treated as 1)
         * @param rng_seed Deterministic RNG seed
         * @param rng_offset Per-sample RNG offset/counter
         * @param device_id Device where data resides
         * @param out_token Host pointer for selected token
         * @param stream Explicit GPU stream
         * @return true if sampled on device
         */
        virtual bool sampleTopKTopPF32(const void *data_device, int n,
                                       int top_k, float top_p, float temperature,
                                       uint64_t rng_seed, uint64_t rng_offset,
                                       int device_id, int *out_token,
                                       void *stream)
        {
            (void)data_device;
            (void)n;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)rng_seed;
            (void)rng_offset;
            (void)device_id;
            (void)out_token;
            (void)stream;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable top-k/top-p/temperature sampling.
         *
         * The selected token is written to out_token_device. This method performs
         * no allocation, host/device copies, or synchronization, and requires an
         * explicit non-null stream.
         */
        virtual bool enqueueSampleTopKTopPF32Device(const void *data_device, int n,
                                                    int top_k, float top_p, float temperature,
                                                    uint64_t rng_seed, uint64_t rng_offset,
                                                    int device_id, void *stream,
                                                    void *out_token_device)
        {
            (void)data_device;
            (void)n;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)rng_seed;
            (void)rng_offset;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            return false;
        }

        /**
         * @brief Publish one host-selected control token into device memory.
         *
         * Some request policies, such as a bounded-thinking stop sequence, choose
         * a token on the host rather than from model logits. GPU execution still
         * requires that token to acquire a persistent device owner before any
         * shifted-MTP or main-graph consumer observes it. Implementations launch
         * a one-thread scalar publication kernel so the scalar is captured in
         * launch parameters; they must not issue an H2D copy from the caller's
         * short-lived stack storage.
         *
         * This operation performs no allocation, transfer API call, default-stream
         * work, or synchronization. The caller owns event publication after this
         * enqueue so every later consumer waits on the exact producer stream.
         *
         * @param value Host request-policy scalar to publish.
         * @param out_value_device Persistent INT32 device destination.
         * @param device_id Backend-local GPU ordinal.
         * @param stream Explicit non-null producer stream.
         * @return true when publication was enqueued successfully.
         */
        virtual bool enqueuePublishInt32ControlScalarDevice(
            int32_t value,
            void *out_value_device,
            int device_id,
            void *stream)
        {
            (void)value;
            (void)out_value_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable top-k/top-p distribution construction.
         *
         * Writes a compact probability table of length top_k to device buffers.
         * Entries outside the selected top-p nucleus are marked with token id -1
         * and probability 0. This method performs no allocation, host/device
         * copies, or synchronization, and requires an explicit non-null stream.
         */
        virtual bool enqueueBuildTopKTopPDistributionF32Device(const void *data_device, int n,
                                                               int top_k, float top_p, float temperature,
                                                               int device_id, void *stream,
                                                               void *out_token_ids_device,
                                                               void *out_probs_device,
                                                               void *scratch_values_device = nullptr,
                                                               void *scratch_indices_device = nullptr,
                                                               int scratch_capacity = 0)
        {
            (void)data_device;
            (void)n;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)device_id;
            (void)stream;
            (void)out_token_ids_device;
            (void)out_probs_device;
            (void)scratch_values_device;
            (void)scratch_indices_device;
            (void)scratch_capacity;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable top-k/top-p distribution construction for rows.
         *
         * Builds `row_count` compact probability tables from a contiguous logits
         * matrix without allocating, synchronizing, or touching the default GPU
         * stream. `row_stride` is measured in FP32 elements between input rows;
         * `out_stride` is measured in INT32/FP32 entries between compact output
         * slots. Scratch capacity is the total number of `(value,index)` entries
         * available across every row in the batched launch.
         *
         * This is the vLLM-style target/bonus verifier-row companion to the
         * scalar distribution builder above. It lets the runner queue all
         * all-position verifier target rows behind one explicit stream handoff
         * instead of launching a scalar table builder per row.
         * @param active_rows_device Optional resident INT32 logical row count;
         *        inactive physical suffix rows perform no vocabulary work.
         */
        virtual bool enqueueBuildTopKTopPDistributionsF32Device(
            const void *data_device,
            int row_count,
            int n,
            int row_stride,
            int top_k,
            float top_p,
            float temperature,
            int device_id,
            void *stream,
            void *out_token_ids_device,
            int out_stride,
            void *out_probs_device,
            void *scratch_values_device = nullptr,
            void *scratch_indices_device = nullptr,
            int scratch_capacity = 0,
            const void *active_rows_device = nullptr)
        {
            (void)data_device;
            (void)row_count;
            (void)n;
            (void)row_stride;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)device_id;
            (void)stream;
            (void)out_token_ids_device;
            (void)out_stride;
            (void)out_probs_device;
            (void)scratch_values_device;
            (void)scratch_indices_device;
            (void)scratch_capacity;
            (void)active_rows_device;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable top-k/top-p processing into full logits.
         *
         * Converts raw logits into the processed-logit representation consumed by
         * the vLLM-style stochastic verifier: temperature is applied, tokens
         * outside the top-k/top-p nucleus are set to -inf, and active tokens keep
         * logits whose softmax exactly matches the compact top-k/top-p
         * distribution. `out_logits_device` may alias `data_device` after the
         * implementation has computed top-k partials, which lets graph stages
         * process LM-head output in place when ownership permits.
         *
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a default/null GPU stream.
         */
        virtual bool enqueueBuildTopKTopPProcessedLogitsF32Device(
            const void *data_device,
            int row_count,
            int n,
            int row_stride,
            int top_k,
            float top_p,
            float temperature,
            int device_id,
            void *stream,
            void *out_logits_device,
            int out_row_stride,
            void *scratch_values_device = nullptr,
            void *scratch_indices_device = nullptr,
            int scratch_capacity = 0)
        {
            (void)data_device;
            (void)row_count;
            (void)n;
            (void)row_stride;
            (void)top_k;
            (void)top_p;
            (void)temperature;
            (void)device_id;
            (void)stream;
            (void)out_logits_device;
            (void)out_row_stride;
            (void)scratch_values_device;
            (void)scratch_indices_device;
            (void)scratch_capacity;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable sampling from a compact probability table.
         *
         * The threshold is a host-provided random draw in [0, 1), allowing callers
         * to preserve their existing deterministic RNG stream while keeping logits
         * and distribution math on the device. The output token is written to a
         * scalar device buffer. When @p threshold_position_device is non-null,
         * the kernel instead derives the draw from @p threshold_seed and
         * `*threshold_position_device + threshold_position_offset`; this is the
         * fully resident MTP path and @p threshold is ignored. Requires an
         * explicit non-null stream.
         */
        virtual bool enqueueSampleDistributionF32Device(
            const void *token_ids_device,
            const void *probs_device,
            int top_k,
            float threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr,
            uint64_t threshold_seed = 0,
            const void *threshold_position_device = nullptr,
            int threshold_position_offset = 0)
        {
            (void)token_ids_device;
            (void)probs_device;
            (void)top_k;
            (void)threshold;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_probability_device;
            (void)threshold_seed;
            (void)threshold_position_device;
            (void)threshold_position_offset;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable sampling from processed full logits.
         *
         * The logits row is already in sampling space: temperature, penalties,
         * and any token masks have been applied. This is the vLLM-style
         * full-logit companion to enqueueSampleDistributionF32Device(), used for
         * bonus-ready or residual rows without materializing compact top-k
         * tables. Implementations must use the explicit non-null stream only.
         */
        virtual bool enqueueSampleProcessedLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr)
        {
            (void)logits_device;
            (void)vocab_size;
            (void)row_stride;
            (void)threshold;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_probability_device;
            return false;
        }

        /**
         * @brief Lazily sample a processed bonus row only when a batch needs it.
         *
         * Stochastic MTP only consumes the bonus ready token when the first
         * target token does not stop and every verified speculative row
         * accepts. This graph-capturable primitive checks the compact verifier
         * outputs on device, returns `-1` immediately when the bonus is not
         * semantically needed, and otherwise samples @p logits_device exactly
         * like enqueueSampleProcessedLogitsF32Device().
         *
         * Backends must use the explicit non-null @p stream only. They must not
         * allocate, synchronize, or fall back to a default/null GPU stream.
         * A non-null @p threshold_position_device selects the same resident
         * position-keyed draw contract as enqueueSampleDistributionF32Device().
         */
        virtual bool enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float threshold,
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            int first_token,
            const void *first_token_device,
            const int *stop_tokens_host,
            int stop_token_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr,
            uint64_t threshold_seed = 0,
            const void *threshold_position_device = nullptr,
            int threshold_position_offset = 0)
        {
            (void)logits_device;
            (void)vocab_size;
            (void)row_stride;
            (void)threshold;
            (void)verify_tokens_device;
            (void)verify_accepted_device;
            (void)row_count;
            (void)first_token;
            (void)first_token_device;
            (void)stop_tokens_host;
            (void)stop_token_count;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_probability_device;
            (void)threshold_seed;
            (void)threshold_position_device;
            (void)threshold_position_offset;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style draft proposal from raw logits.
         *
         * Draft proposal deliberately uses only temperature-scaled raw logits:
         * target rows own top-k/top-p, penalties, and residual correction. The
         * kernel writes the full proposal probability row plus the sampled draft
         * token and q(sampled_token), giving the verifier everything it needs
         * without building a compact top-k/top-p draft table.
         *
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a device-default/null stream.
         */
        virtual bool enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float temperature,
            float threshold,
            int device_id,
            void *stream,
            void *out_probabilities_device,
            int out_row_stride,
            void *out_token_device,
            void *out_probability_device = nullptr)
        {
            (void)logits_device;
            (void)vocab_size;
            (void)row_stride;
            (void)temperature;
            (void)threshold;
            (void)device_id;
            (void)stream;
            (void)out_probabilities_device;
            (void)out_row_stride;
            (void)out_token_device;
            (void)out_probability_device;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style draft proposal while preserving draft logits.
         *
         * This is the production draft-side companion to
         * enqueueSoftmaxAndSampleTemperatureLogitsF32Device(). It samples from
         * the temperature-only proposal distribution, writes the sampled token
         * plus q(sampled_token), and stores the temperature-scaled proposal
         * logits in `out_logits_device` for rejection verification.
         *
         * The verifier can then compute p/q and recovered-token weights from
         * logits/logsumexp, avoiding a full-vocab draft probability matrix.
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a device-default/null stream.
         */
        virtual bool enqueueScaleAndSampleTemperatureLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float temperature,
            float threshold,
            int device_id,
            void *stream,
            void *out_logits_device,
            int out_row_stride,
            void *out_token_device,
            void *out_probability_device = nullptr)
        {
            (void)logits_device;
            (void)vocab_size;
            (void)row_stride;
            (void)temperature;
            (void)threshold;
            (void)device_id;
            (void)stream;
            (void)out_logits_device;
            (void)out_row_stride;
            (void)out_token_device;
            (void)out_probability_device;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable softmax for processed full-logit rows.
         *
         * The input rows are already in sampling space: temperature, penalties,
         * and token masks have been applied. Non-finite logits are written as
         * probability zero. This is the vLLM-style materialization step that
         * feeds full-probability rejection sampling.
         *
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a device-default/null stream.
         */
        virtual bool enqueueSoftmaxProcessedLogitsF32Device(
            const void *logits_device,
            int row_count,
            int vocab_size,
            int row_stride,
            int device_id,
            void *stream,
            void *out_probabilities_device,
            int out_row_stride)
        {
            (void)logits_device;
            (void)row_count;
            (void)vocab_size;
            (void)row_stride;
            (void)device_id;
            (void)stream;
            (void)out_probabilities_device;
            (void)out_row_stride;
            return false;
        }

        /**
         * @brief Fill vLLM-style inverse-exponential rejection samples on device.
         *
         * Writes `row_count` full-vocab rows. Row `r` uses logical position
         * `first_logical_position + r` so the same speculative verification
         * step produces identical recovered-token choices whether it is captured
         * or replayed. Implementations must only enqueue work on the explicit
         * non-null stream; no allocation, synchronization, or default/null
         * stream use is allowed.
         */
        virtual bool enqueueFillInverseExponentialSamplesF32Device(
            void *out_samples_device,
            int row_count,
            int vocab_size,
            int row_stride,
            uint64_t seed,
            int first_logical_position,
            int device_id,
            void *stream)
        {
            (void)out_samples_device;
            (void)row_count;
            (void)vocab_size;
            (void)row_stride;
            (void)seed;
            (void)first_logical_position;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable speculative verify from compact distributions.
         *
         * target/draft distributions must be the top-k probability tables
         * produced by enqueueBuildTopKTopPDistributionF32Device(). The kernel
         * accepts draft_token with min(1, p/q), otherwise samples from the
         * residual max(p - q, 0). It writes only small scalar outputs to device
         * buffers and requires an explicit non-null stream.
         */
        virtual bool enqueueSpeculativeVerifyDistributionsF32Device(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int draft_token,
            uint64_t accept_seed,
            uint64_t accept_offset,
            uint64_t residual_seed,
            uint64_t residual_offset,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)draft_token_ids_device;
            (void)draft_probs_device;
            (void)top_k;
            (void)draft_token;
            (void)accept_seed;
            (void)accept_offset;
            (void)residual_seed;
            (void)residual_offset;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            return false;
        }

        /**
         * @brief Enqueue graph-capturable speculative verify using caller RNG draws.
         *
         * Equivalent to enqueueSpeculativeVerifyDistributionsF32Device(), but
         * consumes explicit accept/residual thresholds instead of deriving random
         * numbers from seed/offset pairs.
         */
        virtual bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)draft_token_ids_device;
            (void)draft_probs_device;
            (void)top_k;
            (void)draft_token;
            (void)accept_threshold;
            (void)residual_threshold;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            return false;
        }

        /**
         * @brief Enqueue batched speculative verification using caller RNG draws.
         *
         * This checks accept/reject decisions and computes each row's candidate
         * residual correction token for several contiguous target/draft
         * distribution slots in one launch. Callers still decide which first
         * rejected row is semantically consumed.
         */
        virtual bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int distribution_stride,
            const int *draft_tokens_host,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int row_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)draft_token_ids_device;
            (void)draft_probs_device;
            (void)top_k;
            (void)distribution_stride;
            (void)draft_tokens_host;
            (void)accept_thresholds_host;
            (void)residual_thresholds_host;
            (void)row_count;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            return false;
        }

        /**
         * @brief Enqueue batched speculative verification using device draft tokens.
         *
         * This is the vLLM-style sibling of
         * enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(): the
         * sampled draft token sequence already lives in arena-owned device
         * memory, so the verifier kernel reads `draft_tokens_device[row]`
         * directly instead of receiving draft tokens as host scalar kernel
         * arguments. If `draft_token_probabilities_device` is provided, row `r`
         * contains q(draft_tokens_device[r]) from the draft sampler and the
         * verifier can skip the sampled-token lookup in the compact draft
         * table. Residual sampling still uses the full draft table.
         *
         * Passing both draft distribution pointers as null is a separate,
         * intentional vLLM-style greedy-draft mode: the draft proposal is
         * treated as one-hot at `draft_tokens_device[row]`, so the verifier can
         * use the compact target distribution without materializing any draft
         * probability table. Passing only one null draft pointer is invalid.
         *
         * Thresholds normally arrive as scalar host values.  For deterministic
         * seeded vLLM-style one-hot verification, callers may pass both
         * threshold arrays as null and provide `inverse_sample_seed` plus
         * exactly one logical-position source. A non-negative
         * `inverse_sample_first_logical_position` is the compatibility scalar
         * source. A non-null `threshold_base_position_device` is the fully
         * resident source; row zero uses the pointed-to value plus
         * `threshold_position_offset`, and later rows increment from there.
         * The backend derives accept/residual thresholds inside the explicit
         * stream launch with `sampling_math::mtp_spec_threshold_from_seed()`.
         * Passing only one null threshold array, omitting the seed, supplying
         * both position sources, or requesting seeded thresholds with a
         * materialized draft distribution is invalid.
         *
         * Implementations must only enqueue work on `stream`; they must not
         * allocate, synchronize, or use a default/null GPU stream.
         */
        virtual bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int distribution_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int row_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            int inverse_sample_vocab_size = 0,
            const void *threshold_base_position_device = nullptr,
            int threshold_position_offset = 0)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)draft_token_ids_device;
            (void)draft_probs_device;
            (void)top_k;
            (void)distribution_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)residual_thresholds_host;
            (void)row_count;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)draft_token_probabilities_device;
            (void)inverse_sample_seed;
            (void)inverse_sample_first_logical_position;
            (void)inverse_sample_vocab_size;
            (void)threshold_base_position_device;
            (void)threshold_position_offset;
            return false;
        }

        /**
         * @brief Enqueue batched stochastic verification from processed full logits.
         *
         * This is the vLLM-style sibling of the compact-table verifier. The
         * target and draft rows are already processed into sampling space
         * (temperature/penalties/masks applied), and the kernel recovers only
         * the sampled draft token probabilities needed for accept/reject. On a
         * rejection it samples the residual distribution from the full logits.
         *
         * Implementations must launch only on the explicit non-null `stream`;
         * no allocation, synchronization, or device-default/null stream use is
         * allowed.
         */
        virtual bool enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_logits_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr)
        {
            (void)target_logits_device;
            (void)draft_logits_device;
            (void)row_count;
            (void)vocab_size;
            (void)target_row_stride;
            (void)draft_row_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)residual_thresholds_host;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)draft_token_probabilities_device;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style verify from target logits and draft probabilities.
         *
         * The target rows are processed logits: penalties, top-k/top-p masks,
         * and temperature have already been applied. The draft rows are
         * temperature-only proposal probabilities captured when each MTP draft
         * token was sampled. In vLLM-style greedy-draft mode
         * @p no_draft_probabilities treats the draft distribution as one-hot at
         * the sampled token. The kernel computes p(draft) from the target row,
         * reads or synthesizes q(draft), and on rejection samples the recovered
         * token by reducing `max(p - q, 0) * inverse_exp(token)`.
         *
         * This keeps the production path closer to vLLM by avoiding target
         * full-probability rows and inverse-random matrices. Implementations
         * must launch only on the explicit non-null `stream`; no allocation,
         * synchronization, or device-default/null stream use is allowed. When
         * @p accept_thresholds_host is null, @p inverse_sample_seed must be
         * non-zero and exactly one position source must be supplied. The
         * resident source is @p threshold_base_position_device plus
         * @p threshold_position_offset; it controls both acceptance and inverse
         * recovery so those two decisions cannot observe different positions.
         */
        virtual bool enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_probabilities_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            bool no_draft_probabilities = false,
            const void *threshold_base_position_device = nullptr,
            int threshold_position_offset = 0)
        {
            (void)target_logits_device;
            (void)draft_probabilities_device;
            (void)row_count;
            (void)vocab_size;
            (void)target_row_stride;
            (void)draft_row_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)inverse_sample_seed;
            (void)inverse_sample_first_logical_position;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)no_draft_probabilities;
            (void)threshold_base_position_device;
            (void)threshold_position_offset;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style verify from target and draft logits.
         *
         * Target rows are processed logits: penalties, top-k/top-p masks, and
         * temperature have already been applied. Draft rows are temperature-only
         * proposal logits captured when each MTP draft token was sampled. The
         * kernel computes p(draft) and q(draft) from row-local logsumexp and, on
         * rejection, samples the recovered token with the same inverse-exp race
         * used by vLLM-style probability rejection.
         *
         * This is a measured alternative to the probability-row path. It avoids
         * materializing full draft probability rows while preserving exact
         * rejection semantics, but production promotion still depends on
         * backend-specific perf evidence. Implementations must launch only on
         * the explicit non-null `stream`; no allocation, synchronization, or
         * default/null stream use is allowed.
         */
        virtual bool enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_logits_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr)
        {
            (void)target_logits_device;
            (void)draft_logits_device;
            (void)row_count;
            (void)vocab_size;
            (void)target_row_stride;
            (void)draft_row_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)inverse_sample_seed;
            (void)inverse_sample_first_logical_position;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)draft_token_probabilities_device;
            return false;
        }

        /**
         * @brief Enqueue vLLM-style stochastic verify from full probability rows.
         *
         * `target_probabilities_device` and `draft_probabilities_device` are
         * full-vocab probability rows for each verifier row. On rejection, the
         * kernel samples the recovered token by reducing
         * `max(target_prob - draft_prob, 0) * inverse_rejection_samples`.
         *
         * This mirrors vLLM's random rejection sampler and is the target
         * production contract for stochastic MTP. Implementations must launch
         * only on the explicit non-null `stream`; no allocation,
         * synchronization, or device-default/null stream use is allowed.
         */
        virtual bool enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
            const void *target_probabilities_device,
            const void *draft_probabilities_device,
            const void *inverse_rejection_samples_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            int inverse_sample_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            bool no_draft_probabilities = false)
        {
            (void)target_probabilities_device;
            (void)draft_probabilities_device;
            (void)inverse_rejection_samples_device;
            (void)row_count;
            (void)vocab_size;
            (void)target_row_stride;
            (void)draft_row_stride;
            (void)inverse_sample_row_stride;
            (void)draft_tokens_device;
            (void)accept_thresholds_host;
            (void)device_id;
            (void)stream;
            (void)out_token_device;
            (void)out_accepted_device;
            (void)out_accept_probability_device;
            (void)out_accept_threshold_device;
            (void)no_draft_probabilities;
            return false;
        }

        /**
         * @brief Enqueue device-side reduction of batched speculative verifier rows.
         *
         * The row verifier writes `verify_tokens_device` and
         * `verify_accepted_device`. This graph-capturable reducer converts those
         * row-local decisions into the vLLM-style output contract: committed
         * output tokens plus a small integer metadata table. Stop tokens are
         * host-side scalars copied into kernel arguments by backend wrappers;
         * the kernel never dereferences host memory. Requires an explicit
         * non-null stream. `out_token_capacity` is the request stride and is
         * initialized in full so unused output slots remain deterministic.
         */
        virtual bool enqueueSummarizeSpeculativeVerifyBatch(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            int first_token,
            const int *stop_tokens_host,
            int stop_token_count,
            const void *bonus_token_device,
            bool has_bonus_token,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device = nullptr,
            int leading_committed_output_count = 0)
        {
            (void)verify_tokens_device;
            (void)verify_accepted_device;
            (void)row_count;
            (void)first_token;
            (void)stop_tokens_host;
            (void)stop_token_count;
            (void)bonus_token_device;
            (void)has_bonus_token;
            (void)device_id;
            (void)stream;
            (void)out_token_capacity;
            (void)out_tokens_device;
            (void)out_meta_device;
            (void)max_state_commit_rows_device;
            (void)leading_committed_output_count;
            return false;
        }

        /**
         * @brief Enqueue batch summary using a device-resident first token.
         *
         * This is the graph-friendly companion to
         * enqueueSummarizeSpeculativeVerifyBatch(). The first main-model token
         * is sampled on GPU and remains in arena scratch until this reducer
         * consumes it. Backends must launch on the explicit `stream`, read
         * `first_token_device` inside the kernel, and call the same shared
         * SamplingMath reducer as the host-token variant.
         *
         * @param first_token_device Device pointer to one INT32 sampled token.
         * @return true when the reducer launch was queued successfully.
         */
        virtual bool enqueueSummarizeSpeculativeVerifyBatchDeviceFirstToken(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            const void *first_token_device,
            const int *stop_tokens_host,
            int stop_token_count,
            const void *bonus_token_device,
            bool has_bonus_token,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device = nullptr,
            int leading_committed_output_count = 0)
        {
            (void)verify_tokens_device;
            (void)verify_accepted_device;
            (void)row_count;
            (void)first_token_device;
            (void)stop_tokens_host;
            (void)stop_token_count;
            (void)bonus_token_device;
            (void)has_bonus_token;
            (void)device_id;
            (void)stream;
            (void)out_token_capacity;
            (void)out_tokens_device;
            (void)out_meta_device;
            (void)max_state_commit_rows_device;
            (void)leading_committed_output_count;
            return false;
        }

        /**
         * @brief Reduce one resident generation transaction from device controls.
         *
         * This is the production stochastic-MTP reducer contract.  Every value
         * that may change after request admission is read from a stable device
         * address on @p stream:
         *
         * - @p first_token_device owns the verifier condition token;
         * - @p stop_tokens_device owns the fixed-width, `-1`-padded stop policy;
         * - @p generation_control_device owns both the current commit budget and
         *   whether row zero was already emitted by the preceding transaction.
         *
         * Exactly one of @p verify_accepted_device and @p greedy_draft_tokens_device
         * must be non-null.  The former reduces ordinary probability-rejection
         * decisions.  The latter compares already-sampled target tokens with the
         * verifier input row, preserving serial-sampling byte equivalence.  A
         * backend implementation must only enqueue work: no allocation, transfer,
         * device synchronization, or host inspection is permitted.
         *
         * @param verify_tokens_device Device token selected for every verifier row.
         * @param verify_accepted_device Device acceptance flag for every row, or
         *        nullptr for serial-equivalent token comparison.
         * @param greedy_draft_tokens_device Device verifier input row used for
         *        serial-equivalent comparison, or nullptr for probability rejection.
         * @param row_count Number of speculative rows in the transaction.
         * @param first_token_device Device pointer to the transaction condition token.
         * @param stop_tokens_device Fixed-width device stop-token row.
         * @param bonus_token_device Optional device terminal sample.
         * @param has_bonus_token Whether @p bonus_token_device is present.
         * @param generation_control_device DeviceGenerationControlIndex row for the
         *        same logical request.
         * @param device_id GPU ordinal.
         * @param stream Exact non-null producer stream.
         * @param out_token_capacity Capacity of the compact output token row.
         * @param out_tokens_device Compact output token destination.
         * @param out_meta_device Compact metadata destination.
         * @return true only when the graph-capturable reducer launch was enqueued.
         */
        virtual bool
        enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            const void *greedy_draft_tokens_device,
            int row_count,
            const void *first_token_device,
            const void *stop_tokens_device,
            const void *bonus_token_device,
            bool has_bonus_token,
            const void *generation_control_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device)
        {
            (void)verify_tokens_device;
            (void)verify_accepted_device;
            (void)greedy_draft_tokens_device;
            (void)row_count;
            (void)first_token_device;
            (void)stop_tokens_device;
            (void)bonus_token_device;
            (void)has_bonus_token;
            (void)generation_control_device;
            (void)device_id;
            (void)stream;
            (void)out_token_capacity;
            (void)out_tokens_device;
            (void)out_meta_device;
            return false;
        }

        /**
         * @brief Sample every seeded serial-equivalent verifier row and reduce it.
         *
         * The compact target matrix contains @p row_count comparison rows followed
         * by one bonus row.  Each row is sampled with the same scalar accumulation
         * order as enqueueSampleDistributionF32Device(), using the draw at
         * `*threshold_position_device + threshold_position_offset + row`.  The
         * sampled comparison rows are then compared byte-for-byte with verifier
         * input entries `[1, row_count]`, while entry zero supplies the already
         * materialized first token.
         *
         * This fused operation removes a fan-out of up to sixteen tiny sampler
         * launches plus a separate summary launch from the captured MTP loop.  It
         * must enqueue exactly one graph-capturable kernel on @p stream.  Dynamic
         * allocation, host/device transfer, atomics, callbacks, and device or
         * stream synchronization are forbidden.
         *
         * @param target_token_ids_device Compact target token rows.
         * @param target_probs_device Compact target probability rows.
         * @param target_row_stride Entries between compact target rows.
         * @param top_k Number of entries inspected in each compact row.
         * @param row_count Physical speculative-comparison capacity captured by
         *        the graph. The resident generation controller selects a logical
         *        depth in `[1, row_count]`; one additional target row at that
         *        logical depth supplies the bonus sample. This distinction lets
         *        one maximum-capacity executable serve every MTP depth without
         *        recapture or a host-authored launch parameter.
         * @param threshold_seed Immutable request sampling seed.
         * @param threshold_position_device Resident pre-verifier base position.
         * @param threshold_position_offset Logical offset of comparison row zero.
         * @param verifier_input_tokens_device Materialized `[first, drafts...]` row.
         * @param stop_tokens_device Fixed-width, `-1`-padded stop-token row.
         * @param generation_control_device Resident transaction budget/control row.
         * @param sampled_target_tokens_device Persistent sampled-row destination.
         * @param out_tokens_device Compact committed-token destination.
         * @param out_meta_device Compact speculative metadata destination.
         * @param first_transaction_diagnostic_device Optional backend-neutral
         *        pointer to a graph-stable
         *        `MTPFirstTransactionDiagnosticRecord`. A non-null pointer
         *        selects the diagnostic kernel specialization; production
         *        callers leave it null and pay no row-hash/copy cost.
         */
        virtual bool
        enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
            const void *target_token_ids_device,
            const void *target_probs_device,
            int target_row_stride,
            int top_k,
            int row_count,
            uint64_t threshold_seed,
            const void *threshold_position_device,
            int threshold_position_offset,
            const void *verifier_input_tokens_device,
            const void *stop_tokens_device,
            const void *generation_control_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *sampled_target_tokens_device,
            void *out_tokens_device,
            void *out_meta_device,
            void *first_transaction_diagnostic_device = nullptr)
        {
            (void)target_token_ids_device;
            (void)target_probs_device;
            (void)target_row_stride;
            (void)top_k;
            (void)row_count;
            (void)threshold_seed;
            (void)threshold_position_device;
            (void)threshold_position_offset;
            (void)verifier_input_tokens_device;
            (void)stop_tokens_device;
            (void)generation_control_device;
            (void)device_id;
            (void)stream;
            (void)out_token_capacity;
            (void)sampled_target_tokens_device;
            (void)out_tokens_device;
            (void)out_meta_device;
            (void)first_transaction_diagnostic_device;
            return false;
        }

        /**
         * @brief Enqueue device-side reduction of greedy verifier rows.
         *
         * Greedy MTP uses the same vLLM-style compact batch contract as the
         * stochastic verifier, but row acceptance is simply
         * `verify_tokens[row] == draft_tokens[row + 1]`. Backends must launch a
         * small graph-capturable kernel on the explicit non-null `stream`,
         * compare the device-resident verifier argmax rows with the
         * device-resident compact verifier input row, and write only compact
         * output tokens plus metadata for the host handoff.
         *
         * @param verify_tokens_device INT32 verifier argmax rows
         *        `[compare_row_count + 1]`; the final row is the bonus ready
         *        token used only when all speculative rows accept.
         * @param draft_tokens_device INT32 verifier input row
         *        `[first_token, draft_1, ...]`.
         * @param compare_row_count Maximum physical speculative rows captured.
         * @param active_verifier_row_count_device Device INT32 logical verifier
         *        rows. The reducer compares exactly this value minus one and
         *        ignores every physical suffix row.
         * @param first_token Legacy host shadow of `draft_tokens_device[0]`.
         *        GPU reducers must read entry zero from @p draft_tokens_device
         *        so deferred first-token paths do not need a pre-verifier D2H.
         */
        virtual bool enqueueSummarizeGreedySpeculativeVerifyBatch(
            const void *verify_tokens_device,
            const void *draft_tokens_device,
            int compare_row_count,
            int first_token,
            const int *stop_tokens_host,
            int stop_token_count,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device = nullptr,
            int leading_committed_output_count = 0)
        {
            (void)verify_tokens_device;
            (void)draft_tokens_device;
            (void)compare_row_count;
            (void)first_token;
            (void)stop_tokens_host;
            (void)stop_token_count;
            (void)device_id;
            (void)stream;
            (void)out_token_capacity;
            (void)out_tokens_device;
            (void)out_meta_device;
            (void)max_state_commit_rows_device;
            (void)leading_committed_output_count;
            return false;
        }

        /**
         * @brief Reduce greedy verifier rows using device-resident controls.
         *
         * This is the reusable GPU-graph form of
         * enqueueSummarizeGreedySpeculativeVerifyBatch().  Both the first token
         * and the fixed-width stop-token row are read by the kernel from stable
         * device addresses.  The stop row always contains
         * `kSpeculativeBatchMaxStopTokens` entries; unused entries are `-1`, so
         * no mutable host count or scalar is captured in the graph node.
         *
         * @param verify_tokens_device Device argmax token for every verifier row.
         * @param draft_tokens_device Device verifier input row; entry zero is the
         *        first target token and later entries are speculative drafts.
         * @param compare_row_count Number of speculative rows to compare.
         * @param active_verifier_row_count_device Device scalar containing the
         *        logical verifier width for this replay.
         * @param stop_tokens_device Fixed-width INT32 stop-token row on device.
         * @param device_id GPU ordinal.
         * @param stream Exact non-null graph execution stream.
         * @param out_token_capacity Capacity of the compact output token row.
         * @param out_tokens_device Compact output token row.
         * @param out_meta_device Compact SamplingMath metadata row.
         * @param max_state_commit_rows_device Device scalar limiting state rows
         *        committed at the current maintenance boundary.
         * @param next_leading_committed_output_count_device Authoritative device
         *        controller scalar describing whether row zero is an already
         *        committed carry from the preceding transaction.
         * @return true when the graph-capturable reducer was enqueued.
         */
        virtual bool
        enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls(
            const void *verify_tokens_device,
            const void *draft_tokens_device,
            int compare_row_count,
            const void *active_verifier_row_count_device,
            const void *stop_tokens_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device = nullptr,
            const void *next_leading_committed_output_count_device = nullptr)
        {
            (void)verify_tokens_device;
            (void)draft_tokens_device;
            (void)compare_row_count;
            (void)active_verifier_row_count_device;
            (void)stop_tokens_device;
            (void)device_id;
            (void)stream;
            (void)out_token_capacity;
            (void)out_tokens_device;
            (void)out_meta_device;
            (void)max_state_commit_rows_device;
            (void)next_leading_committed_output_count_device;
            return false;
        }

        /**
         * @brief Advance one device-owned maintenance clock from batch metadata.
         *
         * Every request summary is evaluated against the same remaining round
         * budget. This graph-capturable reduction advances the global clock by
         * the largest compact output count in the request batch, matching the
         * lockstep round semantics of serial batched decode. The mutable
         * scalar pointers belong to one persistent controller record, and the
         * final flag prevents maintenance from also applying the ordinary
         * one-round serial boundary advance.
         */
        virtual bool enqueueAdvanceSpeculativeCommitBoundary(
            void *meta_device,
            int request_count,
            int meta_stride,
            void *decode_rounds_committed_device,
            void *decode_rounds_until_maintenance_device,
            void *maintenance_due_device,
            void *decode_boundary_advanced_device,
            int device_id,
            void *stream)
        {
            (void)meta_device;
            (void)request_count;
            (void)meta_stride;
            (void)decode_rounds_committed_device;
            (void)decode_rounds_until_maintenance_device;
            (void)maintenance_due_device;
            (void)decode_boundary_advanced_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Publish one serial decode commit into the device MoE clock.
         *
         * Ordinary one-token decode has no compact MTP metadata from which to
         * derive a commit count. This allocation-free, graph-capturable
         * operation therefore advances the persistent clock by exactly one
         * serial-visible round when the edge was not already published by MTP.
         * An existing valid @p decode_boundary_advanced_device marker makes the
         * operation idempotent, allowing the same conditional transaction to
         * follow both serial and grouped-MTP commits without double counting.
         *
         * Reaching zero publishes `maintenance_due=1` and leaves the advanced
         * marker set. The conditional maintenance child consumes both values.
         * A following acknowledgement fragment retires a non-due marker only
         * after the conditional child had an opportunity to consume it. This is
         * the ordinary-decode counterpart of
         * @ref enqueueAdvanceSpeculativeCommitBoundary and shares its persistent
         * controller fields with MTP.
         *
         * Implementations enqueue exactly one kernel on @p stream. They may not
         * inspect the fields on the host, synchronize, allocate, or substitute
         * an eager implementation.
         *
         * @param decode_rounds_committed_device Mutable total committed rounds.
         * @param decode_rounds_until_maintenance_device Mutable remaining cadence.
         * @param maintenance_due_device Mutable zero/one/poison due word.
         * @param decode_boundary_advanced_device Mutable once-only edge marker.
         * @param device_id Backend-local GPU ordinal.
         * @param stream Exact non-null producer stream.
         * @return true only when the publication kernel was enqueued.
         */
        virtual bool enqueuePublishSerialDecodeCommitBoundary(
            void *decode_rounds_committed_device,
            void *decode_rounds_until_maintenance_device,
            void *maintenance_due_device,
            void *decode_boundary_advanced_device,
            int device_id,
            void *stream)
        {
            (void)decode_rounds_committed_device;
            (void)decode_rounds_until_maintenance_device;
            (void)maintenance_due_device;
            (void)decode_boundary_advanced_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Retire a published decode edge after conditional maintenance.
         *
         * This is the mandatory final fragment of a cadence-gated transaction.
         * If maintenance was not due, it clears the once-only advanced marker
         * so the next serial boundary may advance. If maintenance was due but
         * did not run, the due/advanced pair remains intact and the clock cannot
         * cross that edge. A successful maintenance child has already reset the
         * cadence and marker, making this operation an idempotent no-op.
         *
         * Implementations enqueue one graph-capturable kernel on the exact
         * non-null stream and perform no allocation, transfer, or synchronization.
         *
         * @return true only when the acknowledgement kernel was enqueued.
         */
        virtual bool enqueueAcknowledgeDecodeCommitBoundary(
            void *decode_rounds_until_maintenance_device,
            void *maintenance_due_device,
            void *decode_boundary_advanced_device,
            int device_id,
            void *stream)
        {
            (void)decode_rounds_until_maintenance_device;
            (void)maintenance_due_device;
            (void)decode_boundary_advanced_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Bind a persistent HIP MoE scheduler ticket to one request.
         *
         * Request reset calls this allocation-free kernel after resetting the
         * device controller and before publishing reset completion.  The
         * immutable lifecycle fields are therefore ordered with the same arena
         * generation and cannot be inherited by a later request.  CUDA
         * implements the symmetric primitive even though its native
         * conditional graph does not read the ticket on the host.
         *
         * @param session_epoch Exact non-zero request/session generation.
         * @param workspace_generation Exact non-zero persistent arena generation.
         * @param participant_id Local participant index in the MoE domain.
         * @param participant_count Number of mirrored domain participants.
         * @param ticket_device Persistent ticket destination.
         * @param device_id Backend-local GPU ordinal.
         * @param stream Exact non-null request-reset stream.
         * @return true only when initialization was enqueued.
         */
        virtual bool enqueueInitializeDeviceMoERebalanceDispatchTicket(
            uint64_t session_epoch,
            uint64_t workspace_generation,
            uint32_t participant_id,
            uint32_t participant_count,
            void *ticket_device,
            int device_id,
            void *stream)
        {
            (void)session_epoch;
            (void)workspace_generation;
            (void)participant_id;
            (void)participant_count;
            (void)ticket_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Publish one authenticated device-owned MoE dispatch decision.
         *
         * This graph-capturable one-thread kernel runs immediately after the
         * serial/MTP boundary publisher on the exact maintenance stream.  It
         * copies only cadence and health scalars into the request-initialized
         * ticket.  It never copies histograms, plans, routing state, or expert
         * payloads and never performs a transfer or synchronization itself.
         *
         * @param controller_magic_device Device controller magic word.
         * @param controller_version_device Device controller ABI version.
         * @param controller_error_device First fatal controller error word.
         * @param decode_rounds_committed_device Total committed decode rounds.
         * @param decode_rounds_until_maintenance_device Remaining cadence.
         * @param maintenance_due_device Zero/one due predicate.
         * @param decode_boundary_advanced_device Once-only boundary marker.
         * @param ticket_device Persistent initialized ticket destination.
         * @param device_id Backend-local GPU ordinal.
         * @param stream Exact non-null publisher stream.
         * @return true only when publication was enqueued.
         */
        virtual bool enqueuePublishDeviceMoERebalanceDispatchTicket(
            const void *controller_magic_device,
            const void *controller_version_device,
            const void *controller_error_device,
            const void *decode_rounds_committed_device,
            const void *decode_rounds_until_maintenance_device,
            const void *maintenance_due_device,
            const void *decode_boundary_advanced_device,
            void *ticket_device,
            int device_id,
            void *stream)
        {
            (void)controller_magic_device;
            (void)controller_version_device;
            (void)controller_error_device;
            (void)decode_rounds_committed_device;
            (void)decode_rounds_until_maintenance_device;
            (void)maintenance_due_device;
            (void)decode_boundary_advanced_device;
            (void)ticket_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Initialize persistent response and control rows for GPU generation.
         *
         * Request admission supplies the immutable response budget once.  Every
         * later speculative transaction mutates @p control_device and appends to
         * @p response_tokens_device on explicit streams; no per-transaction host
         * mirror participates in the lifecycle.
         */
        virtual bool enqueueInitializeDeviceGeneration(
            int request_count,
            int max_new_tokens,
            const sampling_math::DeviceGenerationDepthPolicy &depth_policy,
            sampling_math::DeviceGenerationLeadingRowDisposition
                initial_leading_row_disposition,
            int response_token_stride,
            void *response_tokens_device,
            int control_stride,
            void *control_device,
            int device_id,
            void *stream)
        {
            (void)request_count;
            (void)max_new_tokens;
            (void)depth_policy;
            (void)initial_leading_row_disposition;
            (void)response_token_stride;
            (void)response_tokens_device;
            (void)control_stride;
            (void)control_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Initialize the isolated host-scheduler ticket identity.
         *
         * This admission-only operation writes a versioned ticket directly on
         * device and snapshots the newly initialized controller into it.  The
         * host never uploads a ticket image.  Keeping the ticket in separate
         * storage from the controller prevents a HIP graph scheduler from
         * observing or mutating response, sampler, KV, or depth-policy state.
         *
         * Implementations enqueue one allocation-free kernel on the exact
         * non-null stream.  Session and workspace generations are immutable
         * request identity; all scheduling decisions remain device-derived.
         */
        virtual bool enqueueInitializeDeviceGenerationDispatchTicket(
            uint64_t session_epoch,
            uint64_t workspace_generation,
            void *control_device,
            int control_stride,
            int request_count,
            void *dispatch_tickets_device,
            int device_id,
            void *stream)
        {
            (void)session_epoch;
            (void)workspace_generation;
            (void)control_device;
            (void)control_stride;
            (void)request_count;
            (void)dispatch_tickets_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Publish one immutable scheduling snapshot per request row.
         *
         * The graph-captured kernel reads only the authoritative generation
         * controller and the optional MoE maintenance predicate.  It writes the
         * narrow @ref sampling_math::DeviceGenerationDispatchTicket ABI consumed
         * by a HIP host scheduler.  No response token, compact verifier outcome,
         * cache position, or sampler state crosses this interface.
         */
        virtual bool enqueuePublishDeviceGenerationDispatchTickets(
            void *control_device,
            int control_stride,
            int request_count,
            const void *maintenance_due_device,
            void *dispatch_tickets_device,
            int device_id,
            void *stream)
        {
            (void)control_device;
            (void)control_stride;
            (void)request_count;
            (void)maintenance_due_device;
            (void)dispatch_tickets_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Reduce device-owned current-batch LLEP markers into terminal control.
         *
         * The complete MoE runtime table is scanned once after its final
         * producer event. The backend writes the number of layers that moved a
         * payload and the number that executed non-owner rows into the named
         * generation-control fields. Request row zero owns this domain-wide
         * evidence; the same fields in later request rows are cleared.
         *
         * Implementations must enqueue one deterministic, graph-capturable
         * kernel on @p stream. Atomics, allocation, host/device transfer, and
         * stream or device synchronization are forbidden.
         *
         * @param runtime_layers_device Contiguous DeviceMoELayerRuntime table.
         * @param layer_count Number of complete layer records in the table.
         * @param generation_control_device Persistent INT32 control-row base.
         * @param generation_control_stride INT32 words between request rows.
         * @param request_count Number of request rows to initialize.
         * @param device_id GPU ordinal.
         * @param stream Exact non-null consumer stream ordered after producers.
         * @return true only when the publication kernel was enqueued.
         */
        virtual bool
        enqueuePublishMoECurrentBatchLLEPEvidenceToGenerationControl(
            const void *runtime_layers_device,
            int layer_count,
            void *generation_control_device,
            int generation_control_stride,
            int request_count,
            int device_id,
            void *stream)
        {
            (void)runtime_layers_device;
            (void)layer_count;
            (void)generation_control_device;
            (void)generation_control_stride;
            (void)request_count;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Derive the next verifier commit budget from resident request state.
         *
         * The generated scalar is consumed directly by the compact verifier
         * reducer.  It is the minimum of response tokens remaining, verifier
         * graph capacity, and the device MoE maintenance boundary. When a MoE
         * boundary is bound, admission also acknowledges the preceding
         * speculative commit if it did not make maintenance due. A due or
         * poisoned boundary is invalid here: its maintenance transaction must
         * complete before another verifier can be admitted.
         */
        virtual bool enqueuePrepareDeviceGenerationTransactionBudget(
            void *control_device,
            int control_stride,
            int request_count,
            int verifier_row_capacity,
            const void *maintenance_rows_remaining_device,
            const void *maintenance_due_device,
            void *decode_boundary_advanced_device,
            int device_id,
            void *stream)
        {
            (void)control_device;
            (void)control_stride;
            (void)request_count;
            (void)verifier_row_capacity;
            (void)maintenance_rows_remaining_device;
            (void)maintenance_due_device;
            (void)decode_boundary_advanced_device;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Atomically commit response bytes and derive publication metadata.
         *
         * One backend thread owns each request row and executes the shared
         * SamplingMath transaction.  Response-ledger append and accepted-state
         * publication are deliberately one launch: callers cannot publish KV,
         * recurrent, or logical state without committing the corresponding
         * serial-visible response bytes first.  The current transaction budget
         * is read from @p control_device; no host scalar is accepted.
         *
         * The call is allocation-free, graph-capturable, and requires the exact
         * non-null producer stream.  A validation failure publishes `out_ok=0`
         * and terminally invalidates the controller row.
         */
        virtual bool enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
            const void *output_tokens_device,
            int output_token_stride,
            void *meta_device,
            int meta_stride,
            const void *base_cached_tokens_device,
            int request_count,
            int padded_state_rows_per_request,
            void *response_tokens_device,
            int response_token_stride,
            void *control_device,
            int control_stride,
            int device_id,
            void *stream,
            void *out_restore_rows_device,
            void *out_target_cached_tokens_device,
            void *out_accepted_state_counts_device,
            void *out_ok_device,
            void *out_next_condition_tokens_device = nullptr,
            void *out_all_drafts_accepted_flags_device = nullptr,
            void *out_stopped_flags_device = nullptr,
            void *out_next_sidecar_condition_tokens_device = nullptr,
            void *out_next_sidecar_position_ids_device = nullptr,
            void *out_next_verifier_condition_tokens_device = nullptr,
            const void *verifier_input_tokens_device = nullptr,
            int verifier_input_token_stride = 0,
            void *out_committed_verifier_identity_device = nullptr)
        {
            (void)output_tokens_device;
            (void)output_token_stride;
            (void)meta_device;
            (void)meta_stride;
            (void)base_cached_tokens_device;
            (void)request_count;
            (void)padded_state_rows_per_request;
            (void)response_tokens_device;
            (void)response_token_stride;
            (void)control_device;
            (void)control_stride;
            (void)device_id;
            (void)stream;
            (void)out_restore_rows_device;
            (void)out_target_cached_tokens_device;
            (void)out_accepted_state_counts_device;
            (void)out_ok_device;
            (void)out_next_condition_tokens_device;
            (void)out_all_drafts_accepted_flags_device;
            (void)out_stopped_flags_device;
            (void)out_next_sidecar_condition_tokens_device;
            (void)out_next_sidecar_position_ids_device;
            (void)out_next_verifier_condition_tokens_device;
            (void)verifier_input_tokens_device;
            (void)verifier_input_token_stride;
            (void)out_committed_verifier_identity_device;
            return false;
        }

        /**
         * @brief Derive device-resident speculative state publication metadata.
         *
         * The compact verifier reducers write one metadata row per request. This
         * graph-capturable helper converts that compact row into the row index
         * and cache-token count consumed by MTP state publication:
         *
         * - restore row: flattened verifier state row to publish
         * - target cached tokens: base cached tokens plus committed verifier rows
         * - accepted state count: verifier rows committed for the request
         * - all-drafts-accepted and stopped flags: transaction predicates that
         *   let downstream graph-captured consumers decide whether the resident
         *   next-condition token is a continuation candidate without first
         *   materializing the compact outcome on the CPU
         * - ok: 1 when the compact metadata was valid for publication
         *
         * `base_cached_tokens_device` is an INT32 array with one entry per
         * request. All output pointers are INT32 arrays with one entry per
         * request. Backends must launch on the explicit non-null `stream`, never
         * allocate inside this call, and use the shared SamplingMath helper so
         * CUDA, ROCm, and CPU-side tests keep the same off-by-one semantics.
         */
        virtual bool enqueueDeriveSpeculativePublicationMetadata(
            const void *meta_device,
            int meta_stride,
            const void *base_cached_tokens_device,
            int request_count,
            int padded_state_rows_per_request,
            int max_state_commit_rows,
            int device_id,
            void *stream,
            void *out_restore_rows_device,
            void *out_target_cached_tokens_device,
            void *out_accepted_state_counts_device,
            void *out_ok_device,
            void *out_next_condition_tokens_device = nullptr,
            const void *output_tokens_device = nullptr,
            int output_token_stride = 0,
            void *out_all_drafts_accepted_flags_device = nullptr,
            void *out_stopped_flags_device = nullptr,
            void *out_next_verifier_condition_tokens_device = nullptr)
        {
            (void)meta_device;
            (void)meta_stride;
            (void)base_cached_tokens_device;
            (void)request_count;
            (void)padded_state_rows_per_request;
            (void)max_state_commit_rows;
            (void)device_id;
            (void)stream;
            (void)out_restore_rows_device;
            (void)out_target_cached_tokens_device;
            (void)out_accepted_state_counts_device;
            (void)out_ok_device;
            (void)out_next_condition_tokens_device;
            (void)output_tokens_device;
            (void)output_token_stride;
            (void)out_all_drafts_accepted_flags_device;
            (void)out_stopped_flags_device;
            (void)out_next_verifier_condition_tokens_device;
            return false;
        }

        /**
         * @brief Derive shifted MTP KV counts from canonical primary publication.
         *
         * The primary target count and validity row already include response,
         * maintenance, and accepted-prefix policy.  Shifted caches must consume
         * those exact outputs rather than re-evaluating compact metadata with a
         * second scalar commit limit.  This call is allocation-free,
         * graph-capturable, and requires the exact non-null producer stream.
         */
        virtual bool
        enqueueDeriveShiftedSpeculativePublicationMetadataFromPrimary(
            const void *base_cached_tokens_device,
            const void *main_target_cached_tokens_device,
            const void *main_publication_ok_device,
            int request_count,
            int mtp_depth,
            int device_id,
            void *stream,
            void *out_target_cached_tokens_device,
            void *out_accepted_state_counts_device,
            void *out_ok_device)
        {
            (void)base_cached_tokens_device;
            (void)main_target_cached_tokens_device;
            (void)main_publication_ok_device;
            (void)request_count;
            (void)mtp_depth;
            (void)device_id;
            (void)stream;
            (void)out_target_cached_tokens_device;
            (void)out_accepted_state_counts_device;
            (void)out_ok_device;
            return false;
        }

        /**
         * @brief Prepare shifted-MTP catch-up condition tokens from compact outcome metadata.
         *
         * Device-resident LocalTP MTP publication must sometimes materialize the
         * accepted shifted sidecar suffix before publishing shifted KV cache
         * counts.  The accepted-count metadata is intentionally still on device,
         * so callers cannot choose the suffix length on the CPU without
         * reintroducing a D2H planning dependency.  This helper writes exactly
         * @p row_count valid token IDs into @p out_tokens_device.  Rows whose
         * suffix index is publishable copy from @p output_tokens_device starting
         * at @p first_output_token_index; rows beyond the compact accepted-state
         * count receive @p filler_token and are later discarded by device-resident
         * shifted-KV publication.  The same launch expands the request's
         * canonical verifier-base count into contiguous absolute positions in
         * @p out_position_ids_device.  Fusing these writes avoids both a host
         * scalar position and a second tiny metadata kernel before graph replay.
         *
         * Backends must enqueue this work on the explicit @p stream, perform no
         * allocation, and keep the same metadata semantics as SamplingMath's
         * speculative batch reducer.
         */
        virtual bool enqueuePrepareSpeculativeShiftedKVTokens(
            const void *meta_device,
            int meta_stride,
            const void *output_tokens_device,
            int output_token_stride,
            int request_index,
            int first_output_token_index,
            int row_count,
            int32_t filler_token,
            int device_id,
            void *stream,
            void *out_tokens_device,
            const void *base_positions_device,
            int position_offset,
            void *out_position_ids_device)
        {
            (void)meta_device;
            (void)meta_stride;
            (void)output_tokens_device;
            (void)output_token_stride;
            (void)request_index;
            (void)first_output_token_index;
            (void)row_count;
            (void)filler_token;
            (void)device_id;
            (void)stream;
            (void)out_tokens_device;
            (void)base_positions_device;
            (void)position_offset;
            (void)out_position_ids_device;
            return false;
        }

        /**
         * @brief Compose one request-batched MTP sidecar input entirely on GPU.
         *
         * Request-batched MTP stores sampled proposals in request-major slots:
         * depth @c d for request @c r lives at
         * `condition_tokens_device[r * condition_token_stride]` after the caller
         * advances the source pointer to depth @c d. The next grouped sidecar,
         * however, consumes contiguous request rows. This primitive gathers
         * those token IDs and derives each row's absolute RoPE/KV position from
         * a device-owned base-position mailbox in one small launch. Keeping the
         * two values in one preparation kernel gives the captured sidecar stable
         * arena pointers and removes the former D2H token shadow followed by H2D
         * restaging at every speculative depth.
         *
         * Implementations must enqueue on the explicit non-null @p stream,
         * perform no allocation or synchronization, and support the production
         * request-batch bound of one through four rows.
         *
         * @param condition_tokens_device Device INT32 proposal slots.
         * @param condition_token_stride Element stride between request proposals.
         * @param base_positions_device Device INT32 base position for each request.
         * @param position_offset Draft-depth offset added to every base position.
         * @param request_count Number of request rows to compose.
         * @param device_id GPU ordinal owning every pointer.
         * @param stream Explicit CUDA/HIP stream used by the following sidecar.
         * @param out_condition_tokens_device Contiguous INT32 output token rows.
         * @param out_position_ids_device Contiguous INT32 absolute positions.
         * @return true when the preparation kernel was enqueued successfully.
         */
        virtual bool enqueuePrepareMTPBatchedSidecarInputs(
            const void *condition_tokens_device,
            int condition_token_stride,
            const void *base_positions_device,
            int position_offset,
            int request_count,
            int device_id,
            void *stream,
            void *out_condition_tokens_device,
            void *out_position_ids_device)
        {
            (void)condition_tokens_device;
            (void)condition_token_stride;
            (void)base_positions_device;
            (void)position_offset;
            (void)request_count;
            (void)device_id;
            (void)stream;
            (void)out_condition_tokens_device;
            (void)out_position_ids_device;
            return false;
        }

        /**
         * @brief Expand device-owned request positions into grouped verifier rows.
         *
         * A grouped verifier consumes a request-major matrix with shape
         * `[request_count, padded_seq_len]`. The canonical next position for
         * request @c r is stored in @p base_positions_device and can change after
         * every accepted-state publication without any corresponding host
         * update. This primitive writes
         * `out[r * padded_seq_len + token] = base[r] + token`, allowing RoPE and
         * every position-dependent verifier stage to consume the same
         * device-owned logical state as the KV cache.
         *
         * Implementations must enqueue one bounded launch on the explicit
         * non-null @p stream. They must not allocate, synchronize, or copy
         * mutable logical state through the host.
         *
         * @param base_positions_device Device INT32 next-position row, one value
         *        per request.
         * @param request_count Number of independent request rows.
         * @param padded_seq_len Physical verifier columns per request.
         * @param device_id GPU ordinal owning both input and output.
         * @param stream Explicit CUDA/HIP verifier execution stream.
         * @param out_position_ids_device Device INT32 request-major output with
         *        `request_count * padded_seq_len` elements.
         * @return true when the position expansion was enqueued successfully.
         */
        virtual bool enqueuePrepareMTPVerifierPositionIds(
            const void *base_positions_device,
            int request_count,
            int padded_seq_len,
            int device_id,
            void *stream,
            void *out_position_ids_device)
        {
            (void)base_positions_device;
            (void)request_count;
            (void)padded_seq_len;
            (void)device_id;
            (void)stream;
            (void)out_position_ids_device;
            return false;
        }

        /**
         * @brief Materialize all dynamic geometry consumed by a grouped verifier.
         *
         * The grouped verifier owns two related but distinct device inputs:
         * absolute position rows and the number of valid tokens in each padded
         * request row. Position rows derive from the canonical device KV counts.
         * Request lengths derive from the physical valid-row list already
         * published for compact verifier logits. Keeping both outputs in one
         * bounded launch gives RoPE, short-conv, and GDN recurrence one ordered
         * geometry publication without introducing another H2D transfer.
         *
         * When @p valid_graph_rows_device is null, each request owns the same
         * dense logical prefix and its length is
         * `valid_graph_row_count / request_count`. This may be smaller than
         * @p padded_seq_len for a fixed-width scalar verifier bucket. Otherwise
         * the array contains @p valid_graph_row_count flattened request-major
         * physical indices. An implementation must count those indices per
         * request exactly; gaps are padding and must not mutate recurrent state.
         *
         * A device-generation parent supplies @p generation_control_device.
         * In that mode each request length comes from the controller's active
         * verifier-row field, and the kernel must validate that it equals the
         * current draft depth plus one. Invalid controller geometry is a fatal
         * controller error, never a reason to use the static row plan.
         *
         * Implementations must be graph-capturable, allocation-free, transfer-free,
         * and asynchronous on the explicit non-null @p stream.
         *
         * @param base_positions_device Device INT32 next-position row.
         * @param valid_graph_rows_device Optional device INT32 physical valid rows.
         * @param valid_graph_row_count Number of entries in the valid-row array.
         * @param generation_control_device Optional mutable controller rows.
         * @param generation_control_stride Controller words between requests;
         *        zero exactly when @p generation_control_device is null.
         * @param request_count Number of independent request rows.
         * @param padded_seq_len Physical verifier columns per request.
         * @param device_id GPU ordinal owning all inputs and outputs.
         * @param stream Exact CUDA/HIP producer stream.
         * @param out_position_ids_device Device INT32 request-major positions.
         * @param out_request_lengths_device Device INT32 valid width per request.
         * @return true when the geometry publication was enqueued.
         */
        virtual bool enqueuePrepareMTPVerifierGeometry(
            const void *base_positions_device,
            const void *valid_graph_rows_device,
            int valid_graph_row_count,
            void *generation_control_device,
            int generation_control_stride,
            int request_count,
            int padded_seq_len,
            int device_id,
            void *stream,
            void *out_position_ids_device,
            void *out_request_lengths_device)
        {
            (void)base_positions_device;
            (void)valid_graph_rows_device;
            (void)valid_graph_row_count;
            (void)generation_control_device;
            (void)generation_control_stride;
            (void)request_count;
            (void)padded_seq_len;
            (void)device_id;
            (void)stream;
            (void)out_position_ids_device;
            (void)out_request_lengths_device;
            return false;
        }

        /**
         * @brief Fuse one controller-owned verifier token row and its geometry.
         *
         * Device-controlled generation changes draft depth between parent-loop
         * iterations. Capturing memcpy byte counts would freeze that depth in the
         * executable, so this primitive reads the authoritative controller row on
         * device and publishes the complete padded verifier input in one launch:
         * condition token, active draft prefix, zeroed inactive suffix, absolute
         * positions, logical request length, and pre-verifier KV base snapshot.
         * Invalid depth/row relations terminally poison the controller through the
         * shared DeviceGenerationError ABI; they never select a static copy path.
         *
         * Implementations must be graph-capturable, allocation-free, transfer-free,
         * and asynchronous on the exact non-null @p stream.
         *
         * @param first_token_device Device INT32 condition token.
         * @param draft_tokens_device Device INT32 draft slots, with capacity at
         *        least `padded_seq_len - 1`.
         * @param base_position_device Device INT32 live next-position scalar.
         * @param generation_control_row_device First word of the request's
         *        mutable generation-controller row.
         * @param generation_control_stride Number of INT32 words in that row.
         * @param padded_seq_len Physical verifier width captured by this branch.
         * @param device_id GPU ordinal owning every pointer.
         * @param stream Exact producer stream.
         * @param out_tokens_device Device INT32 padded verifier token row.
         * @param out_position_ids_device Device INT32 padded absolute positions.
         * @param out_request_length_device Device INT32 logical row count.
         * @param out_base_position_snapshot_device Device INT32 immutable base
         *        snapshot consumed by accepted-state publication.
         * @return true when the fused publication was enqueued.
         */
        virtual bool enqueuePrepareMTPVerifierControlledRow(
            const void *first_token_device,
            const void *draft_tokens_device,
            const void *base_position_device,
            void *generation_control_row_device,
            int generation_control_stride,
            int padded_seq_len,
            int device_id,
            void *stream,
            void *out_tokens_device,
            void *out_position_ids_device,
            void *out_request_length_device,
            void *out_base_position_snapshot_device)
        {
            (void)first_token_device;
            (void)draft_tokens_device;
            (void)base_position_device;
            (void)generation_control_row_device;
            (void)generation_control_stride;
            (void)padded_seq_len;
            (void)device_id;
            (void)stream;
            (void)out_tokens_device;
            (void)out_position_ids_device;
            (void)out_request_length_device;
            (void)out_base_position_snapshot_device;
            return false;
        }

        /**
         * @brief Seed the request-batched GPU logical-state mailbox after prefill.
         *
         * The terminal prefill sampler already owns one sampled token per request
         * in @p sampled_tokens_device. Request admission has likewise published
         * each prompt's next logical position to @p target_positions_device. This
         * primitive turns those two device rows into the first complete MTP
         * publication mailbox in one bounded kernel launch. No mutable position or
         * sampled-token shadow is adopted from host memory at this boundary.
         *
         * Production request admission and publication use separate persistent
         * arena rows: the immutable admitted prompt lengths are inputs, while
         * the outputs comprise one base-cache scratch row and six
         * request-lifetime logical-state rows. Implementations must nevertheless
         * remain alias-safe for focused kernel tests and must enqueue on the
         * explicit non-null @p stream without allocation, synchronization, or
         * a default-stream substitute.
         *
         * @param sampled_tokens_device Contiguous INT32 sampled prefill tokens.
         * @param target_positions_device Device INT32 prompt positions, one per request.
         * @param request_count Number of initialized request rows (one through four).
         * @param device_id GPU ordinal owning every device pointer.
         * @param stream Explicit CUDA/HIP stream ordered after terminal sampling.
         * @param out_base_cached_tokens_device Device INT32 base cache lengths.
         * @param out_target_positions_device Device INT32 next decode positions.
         * @param out_accepted_state_counts_device Device INT32 zeroed accept counts.
         * @param out_next_condition_tokens_device Device INT32 sampled conditions.
         * @param out_all_drafts_accepted_flags_device Device INT32 zeroed flags.
         * @param out_stopped_flags_device Device INT32 zeroed stop flags.
         * @param out_publication_ok_flags_device Device INT32 one-valued validity flags.
         * @return true when mailbox initialization was enqueued successfully.
         */
        virtual bool enqueueInitializeMTPDeviceLogicalState(
            const void *sampled_tokens_device,
            const void *target_positions_device,
            int request_count,
            int device_id,
            void *stream,
            void *out_base_cached_tokens_device,
            void *out_target_positions_device,
            void *out_accepted_state_counts_device,
            void *out_next_condition_tokens_device,
            void *out_all_drafts_accepted_flags_device,
            void *out_stopped_flags_device,
            void *out_publication_ok_flags_device)
        {
            (void)sampled_tokens_device;
            (void)target_positions_device;
            (void)request_count;
            (void)device_id;
            (void)stream;
            (void)out_base_cached_tokens_device;
            (void)out_target_positions_device;
            (void)out_accepted_state_counts_device;
            (void)out_next_condition_tokens_device;
            (void)out_all_drafts_accepted_flags_device;
            (void)out_stopped_flags_device;
            (void)out_publication_ok_flags_device;
            return false;
        }

        /**
         * @brief GPU-side sparse logit penalty application
         *
         * Applies a sparse set of additive penalties to logits in-place on the GPU.
         * Each entry (token_id, penalty) subtracts the penalty from logits[token_id].
         * Used to apply presence, frequency, and DRY penalties without a full D2H
         * transfer of the logits tensor (~600KB for 151K vocab).
         *
         * @param logits_device Device pointer to FP32 logits [vocab_size] — modified in-place
         * @param token_ids_host Host array of token IDs to penalize
         * @param penalties_host Host array of penalty values (positive = penalize)
         * @param num_penalties Number of entries in token_ids and penalties arrays
         * @param vocab_size Total vocabulary size (for bounds checking)
         * @param device_id Device where logits reside
         * @param stream Optional stream for async execution
         * @return true if executed on device, false if not supported
         */
        virtual bool prepareLogitPenaltyWorkspace(
            int vocab_size,
            int device_id)
        {
            (void)vocab_size;
            (void)device_id;
            return false;
        }

        virtual bool applyLogitPenaltiesF32(void *logits_device,
                                            const int *token_ids_host,
                                            const float *penalties_host,
                                            int num_penalties, int vocab_size,
                                            int device_id, void *stream)
        {
            (void)logits_device;
            (void)token_ids_host;
            (void)penalties_host;
            (void)num_penalties;
            (void)vocab_size;
            (void)device_id;
            (void)stream;
            return false; // Not supported by default
        }

        /**
         * @brief Enqueue sparse logit penalties from device-resident inputs.
         *
         * This is the graph-capturable form of applyLogitPenaltiesF32(): token IDs
         * and penalty values already live on the target device, the caller supplies
         * an explicit non-null stream, and the backend only enqueues the penalty
         * kernel. It performs no allocation, host/device copies, or synchronization.
         *
         * @param logits_device Device pointer to FP32 logits [vocab_size], modified in-place
         * @param token_ids_device Device pointer to int token IDs [num_penalties]
         * @param penalties_device Device pointer to FP32 penalties [num_penalties]
         * @param num_penalties Number of entries in token_ids_device and penalties_device
         * @param vocab_size Total vocabulary size for bounds checking
         * @param device_id Device where all pointers reside
         * @param stream Explicit non-null GPU stream
         * @return true if the kernel launch was enqueued
         */
        virtual bool enqueueLogitPenaltiesF32Device(void *logits_device,
                                                    const void *token_ids_device,
                                                    const void *penalties_device,
                                                    int num_penalties, int vocab_size,
                                                    int device_id, void *stream)
        {
            (void)logits_device;
            (void)token_ids_device;
            (void)penalties_device;
            (void)num_penalties;
            (void)vocab_size;
            (void)device_id;
            (void)stream;
            return false;
        }

        // ====================================================================
        // Compute Operations
        // ====================================================================

        /**
         * @brief Quantized matrix multiplication: C = A * B (IQ4_NL format)
         *
         * Performs GEMM with FP32 activations and IQ4_NL quantized weights.
         *
         * @param A_device Device pointer to FP32 matrix A [m × k] row-major
         * @param B_device Device pointer to IQ4_NL quantized matrix B [n × k/32] blocks
         * @param C_device Device pointer to FP32 output matrix C [m × n] row-major
         * @param m Number of rows in A and C
         * @param n Number of columns in B and C
         * @param k Number of columns in A and rows in B (must be multiple of 32)
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Requirements**:
         * - All pointers must be valid device memory
         * - k must be a multiple of 32 (IQ4_NL block size)
         * - Matrices must be in row-major format
         *
         * **IQ4_NL Format** (each block encodes 32 floats in 18 bytes):
         * - 2 bytes: FP16 scale factor
         * - 16 bytes: Packed 4-bit indices (2 per byte)
         * - Effective: 4.5 bits/value (~7.1× compression vs FP32)
         *
         * **Semantics**:
         * - CUDA: Calls IQ4_NL_Gemm.cu kernel
         * - ROCm: Calls IQ4_NL_Gemm.hip kernel (future)
         * - CPU: Not applicable (use CPU kernel directly)
         */
        virtual bool gemmIQ4NL(
            const void *A_device,
            const void *B_device,
            void *C_device,
            int m,
            int n,
            int k,
            int device_id) = 0;

        // ====================================================================
        // Stream Management
        // ====================================================================

        /**
         * @brief Create a non-blocking stream on the specified device
         *
         * @param device_id GPU device ID (0-based)
         * @return Opaque stream handle (nullptr on failure)
         *
         * **Semantics**:
         * - CUDA: cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
         * - ROCm: hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)
         * - CPU: returns dummy non-null pointer
         *
         * **Lifetime**: Caller owns the stream and must call destroyStream()
         */
        virtual void *createStream(int device_id)
        {
            (void)device_id;
            return nullptr;
        }

        /**
         * @brief Destroy a stream created by createStream()
         *
         * @param stream Opaque stream handle (may be nullptr)
         * @param device_id GPU device ID (0-based)
         */
        virtual void destroyStream(void *stream, int device_id)
        {
            (void)stream;
            (void)device_id;
        }

        /**
         * @brief Synchronize a specific stream (wait for all queued work to complete)
         *
         * @param stream Opaque stream handle. GPU backends require an explicit,
         * non-null stream; CPU backends may ignore it.
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaStreamSynchronize(stream)
         * - ROCm: hipStreamSynchronize(stream)
         * - CPU: no-op (always synchronous)
         */
        virtual bool synchronizeStream(void *stream, int device_id)
        {
            (void)stream;
            (void)device_id;
            return true;
        }

        /**
         * @brief Make a stream wait for an event before proceeding
         *
         * All operations enqueued on the stream after this call will wait until
         * the event is recorded and completed.
         *
         * @param stream Opaque stream handle
         * @param event Opaque event handle
         * @param device_id GPU device ID (0-based)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaStreamWaitEvent(stream, event, 0)
         * - ROCm: hipStreamWaitEvent(stream, event, 0)
         * - CPU: no-op (always synchronous)
         */
        virtual bool streamWaitEvent(void *stream, void *event, int device_id)
        {
            (void)stream;
            (void)event;
            (void)device_id;
            return true;
        }

        /**
         * @brief Report whether exact-stream 32-bit timeline signals are supported.
         *
         * A timeline signal is a device-owned 32-bit word that one explicit GPU
         * stream can wait on while another explicit stream publishes a monotonically
         * increasing value.  Unlike an event, the wait may be enqueued before the
         * corresponding publication exists.  This property is required by
         * background producer/consumer protocols whose host scheduler must return
         * before a worker has prepared the completion operation.
         *
         * Implementations must return false unless all four operations are usable
         * for @p device_id: allocation, destruction, stream wait, and stream
         * publication.  Callers must not substitute a host wait when this capability
         * is absent.
         *
         * @param device_id Backend-local GPU ordinal.
         * @return true only when the complete timeline-signal contract is available.
         */
        virtual bool supportsStreamTimelineSignal32(int device_id) const
        {
            (void)device_id;
            return false;
        }

        /**
         * @brief Allocate one device-owned 32-bit timeline signal.
         *
         * The returned storage is intentionally uninitialized.  Its owner must
         * publish an initial value on an exact stream and prove that publication
         * complete before exposing the signal to consumers.
         *
         * @param device_id Backend-local GPU ordinal.
         * @return Opaque device address, or nullptr on failure/unsupported hardware.
         */
        virtual void *allocateStreamTimelineSignal32(int device_id)
        {
            (void)device_id;
            return nullptr;
        }

        /**
         * @brief Destroy a timeline signal after every referencing stream has drained.
         * @param signal Signal returned by allocateStreamTimelineSignal32(); nullptr is ignored.
         * @param device_id Backend-local GPU ordinal that owns @p signal.
         */
        virtual void freeStreamTimelineSignal32(void *signal, int device_id)
        {
            (void)signal;
            (void)device_id;
        }

        /**
         * @brief Make an exact stream wait for a future or already-published value.
         *
         * Work submitted after this call may execute only after the unsigned signal
         * value is greater than or equal to @p value.  Values must increase
         * monotonically for the signal lifetime; callers own overflow prevention.
         * This method must enqueue only and must never synchronize the host.
         *
         * @param stream Exact non-null consumer stream.
         * @param signal Device signal owned by @p device_id.
         * @param value Monotonic completion generation to await.
         * @param device_id Backend-local GPU ordinal.
         * @return true when the wait was enqueued.
         */
        virtual bool streamWaitTimelineSignal32(
            void *stream,
            void *signal,
            uint32_t value,
            int device_id)
        {
            (void)stream;
            (void)signal;
            (void)value;
            (void)device_id;
            return false;
        }

        /**
         * @brief Publish a monotonically increasing value from an exact stream.
         *
         * The device writes @p value only after all earlier work on @p stream has
         * completed.  The call enqueues the publication and returns without host
         * synchronization.
         *
         * @param stream Exact non-null producer stream.
         * @param signal Device signal owned by @p device_id.
         * @param value Monotonic completion generation to publish.
         * @param device_id Backend-local GPU ordinal.
         * @return true when the publication was enqueued.
         */
        virtual bool streamPublishTimelineSignal32(
            void *stream,
            void *signal,
            uint32_t value,
            int device_id)
        {
            (void)stream;
            (void)signal;
            (void)value;
            (void)device_id;
            return false;
        }

        /**
         * @brief Report whether exact-stream 64-bit timeline waits/writes exist.
         *
         * The address may be backend-owned signal memory or a device alias of
         * portable mapped host pages. Callers must prove the latter separately
         * by registering the region and executing a real producer/consumer
         * integration test; capability advertisement alone is not evidence.
         */
        virtual bool supportsStreamTimelineSignal64(int device_id) const
        {
            (void)device_id;
            return false;
        }

        /** @brief Allocate one backend-owned 64-bit timeline word for device-local use. */
        virtual void *allocateStreamTimelineSignal64(int device_id)
        {
            (void)device_id;
            return nullptr;
        }

        /** @brief Free one backend-owned 64-bit timeline after all streams drain. */
        virtual void freeStreamTimelineSignal64(void *signal, int device_id)
        {
            (void)signal;
            (void)device_id;
        }

        /**
         * @brief Enqueue an unsigned-GEQ wait on an exact non-null stream.
         * @param stream Exact consumer stream; default/null streams are invalid.
         * @param signal GPU-accessible aligned 64-bit word.
         * @param value Monotonically increasing value to await.
         * @param device_id Exact local device ordinal interpreting @p signal.
         */
        virtual bool streamWaitTimelineSignal64(
            void *stream,
            void *signal,
            uint64_t value,
            int device_id)
        {
            (void)stream;
            (void)signal;
            (void)value;
            (void)device_id;
            return false;
        }

        /**
         * @brief Enqueue one fenced 64-bit publication on an exact stream.
         *
         * Every earlier payload write/copy on @p stream must become visible
         * before a peer released by this publication can consume it. The method
         * enqueues only and never synchronizes the host.
         */
        virtual bool streamPublishTimelineSignal64(
            void *stream,
            void *signal,
            uint64_t value,
            int device_id)
        {
            (void)stream;
            (void)signal;
            (void)value;
            (void)device_id;
            return false;
        }

        // ====================================================================
        // Infrastructure copy submission (no implicit host synchronization)
        // ====================================================================

        /**
         * @brief Submit async H2D copy on a specific stream WITHOUT synchronizing
         *
         * TransferEngine uses this backend primitive and immediately publishes
         * an exact completion event into tensor coherence. Captured transfer
         * nodes and explicitly reviewed loader/collective infrastructure may
         * also use it when a graph or batch-level event owns completion. Ordinary
         * production callers must not bypass TransferEngine.
         *
         * @param dst Device destination pointer (must be pre-allocated)
         * @param src Host source pointer (should be pinned for true async DMA)
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @param stream Opaque stream handle (must not be nullptr)
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream)
         * - ROCm: hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream)
         * - CPU: memcpy (synchronous fallback)
         */
        virtual bool hostToDeviceOnStream(void *dst, const void *src, size_t bytes,
                                          int device_id, void *stream)
        {
            // CPU/test backends may complete inline; GPU backends override.
            return hostToDevice(dst, src, bytes, device_id, stream);
        }

        /**
         * @brief Submit async D2H copy on a specific stream WITHOUT synchronizing
         *
         * This is the device-to-host companion to hostToDeviceOnStream().  It is
         * TransferEngine normally follows this enqueue with one exact event and
         * waits that event only at a real host-observation boundary. Captured
         * publication nodes may leave completion to their owning graph event.
         * GPU implementations reject nullptr streams; CPU implementations may
         * treat the stream as an ignored synchronous marker.
         *
         * @param dst Host destination pointer (should be pinned for true async DMA)
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @param stream Opaque stream handle (must not be nullptr for GPU)
         * @return true on successful enqueue/copy, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream)
         * - ROCm: hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream)
         * - CPU: memcpy (synchronous fallback)
         */
        virtual bool deviceToHostOnStream(void *dst, const void *src, size_t bytes,
                                          int device_id, void *stream)
        {
            // Default: fall back to synchronous deviceToHost.
            return deviceToHost(dst, src, bytes, device_id, stream);
        }

        /**
         * @brief Launch a bounded byte copy between device-visible addresses.
         *
         * This primitive exists for progress-sensitive transfers competing
         * with a retained graph's own mapped system-memory traffic. The mapped
         * endpoint is its exact TransferEngine-owned device alias, never a host
         * pointer guessed to share that address. Either direction is supported.
         * Implementations enqueue a byte-exact kernel on @p stream
         * and never allocate, wait, synchronize, or substitute a DMA copy.
         * Ordinary callers use TransferEngine, which validates registration,
         * bounds, endpoint identity, and the exact stream before reaching here.
         *
         * @param dst Stable device or mapped-alias destination bytes.
         * @param src Stable device or mapped-alias source bytes.
         * @param bytes Positive byte count to copy exactly.
         * @param device_id Exact source GPU ordinal.
         * @param stream Exact non-null stream whose event owns publication.
         * @return True only when the kernel launch was accepted.
         */
        virtual bool copyDeviceVisibleRegionByKernelOnStream(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id,
            void *stream)
        {
            (void)dst;
            (void)src;
            (void)bytes;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Resolve both mapped-copy kernels before concurrent execution.
         * @param device_id Exact GPU whose module generation must be prepared.
         * @return True when all required function handles are materialized.
         *
         * TransferEngine invokes this during execution-lane construction, not
         * on submission. Lazy module loading can synchronize a context and
         * strand a first-use kernel behind a peer-held inference graph.
         */
        virtual bool prepareMappedHostCopyKernels(int device_id)
        {
            (void)device_id;
            return false;
        }

        /**
         * @brief Submit a mapped copy with progress independent of peer-held graphs.
         * @param device_region Stable device source or inactive destination.
         * @param mapped_host Registered host address used by native DMA.
         * @param mapped_alias Exact device alias used by device copy kernels.
         * @param bytes Positive validated extent of both regions.
         * @param direction Immutable source/destination ownership direction.
         * @param device_id Exact local GPU ordinal.
         * @param stream Exact prepared background stream, never null.
         * @return True only when the native enqueue succeeds.
         *
         * CUDA copy queues can alias a peer-held graph's pending copy; HIP
         * compute queues can alias its held kernel. Backends therefore own the
         * native progress mechanism (CUDA bounded kernel, HIP asynchronous DMA).
         * This is one contract, not a failed-operation retry or mode fallback.
         * TransferEngine validates ownership and both aliases before calling it.
         */
        virtual bool enqueueBackgroundMappedCopyOnStream(
            void *device_region, void *mapped_host, void *mapped_alias,
            size_t bytes, MappedTransferDirection direction,
            int device_id, void *stream)
        {
            (void)device_region; (void)mapped_host; (void)mapped_alias;
            (void)bytes; (void)direction; (void)device_id; (void)stream;
            return false;
        }

        /**
         * @brief Snapshot fixed mapped transfer commands on an exact stream.
         *
         * One capture-safe block owns each permanent slot and copies its
         * host-published command into ordinary device memory.  This is the
         * primary-stream half of a graph-owned parallel progress branch.  It
         * performs no allocation, wait, callback, synchronization, or payload
         * transfer. Production callers reach it only through TransferEngine.
         *
         * @param commands Device alias of the fixed mapped command array.
         * @param claims Persistent device-resident immutable snapshots.
         * @param slot_capacity Positive immutable command-array cardinality.
         * @param device_id Exact local GPU ordinal.
         * @param stream Exact non-null primary capture stream.
         * @return True only when the claim kernel launch was accepted.
         */
        virtual bool enqueueMappedTransferProgressClaims(
            const MappedTransferProgressCommand *commands,
            MappedTransferProgressClaim *claims,
            size_t slot_capacity,
            int device_id,
            void *stream)
        {
            (void)commands;
            (void)claims;
            (void)slot_capacity;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Copy every claimed mapped transfer on an exact branch stream.
         *
         * One block owns each slot, skips an idle or completed generation, and
         * copies an active process-local source/destination pair before a
         * system-scope release publication. Either endpoint may be VRAM or a
         * TransferEngine-registered mapped host alias. The method is capture-
         * safe and never allocates, waits, synchronizes, or invokes the host.
         *
         * @param claims Device-resident snapshots produced by the claim phase.
         * @param completions Device alias of the fixed mapped result array.
         * @param slot_capacity Positive immutable command-array cardinality.
         * @param maximum_bytes Positive per-command byte bound.
         * @param device_id Exact local GPU ordinal.
         * @param stream Exact non-null auxiliary capture stream.
         * @return True only when the payload kernel launch was accepted.
         */
        virtual bool enqueueMappedTransferProgressCopies(
            const MappedTransferProgressClaim *claims,
            MappedTransferProgressCompletion *completions,
            size_t slot_capacity,
            size_t maximum_bytes,
            int device_id,
            void *stream)
        {
            (void)claims;
            (void)completions;
            (void)slot_capacity;
            (void)maximum_bytes;
            (void)device_id;
            (void)stream;
            return false;
        }

        /**
         * @brief Initialize device-only copy cursors on an exact setup stream.
         * @param cursors Persistent physical-inbox state owned by TransferEngine.
         * @param capacity Positive count of complete cursor records.
         * @param device_id Exact GPU ordinal.
         * @param stream Non-null setup stream, before any service execution.
         * @return Whether initialization was enqueued; unsupported backends reject.
         */
        virtual bool initializeMappedTransferService(
            MappedTransferServiceCursor *cursors, size_t capacity,
            int device_id, void *stream)
        {
            (void)cursors; (void)capacity; (void)device_id; (void)stream;
            return false;
        }

        /**
         * @brief Record a GPU-owned graph interval transition, never a host flag.
         * @param interval Persistent private word belonging to exactly one graph.
         * @param value Typed Open at fork or Closed before join.
         * @param device_id Exact GPU ordinal.
         * @param stream Non-null primary capture stream.
         * @return Whether the lifecycle kernel was accepted.
         */
        virtual bool enqueueMappedTransferInterval(
            std::uint32_t *interval, MappedTransferInterval value,
            int device_id, void *stream)
        {
            (void)interval; (void)value; (void)device_id; (void)stream;
            return false;
        }

        /**
         * @brief Submit a resumable copy worker under its explicit lifetime.
         * @param commands Exact mapped physical-inbox command array.
         * @param completions Exact mapped device-authored receipt array.
         * @param cursors Persistent device claimant/cursor array.
         * @param capacity Physical concurrency bound, not topology slot count.
         * @param maximum_bytes Admission bound for every immutable command.
         * @param interval Graph-private lifecycle word, absent for PublishedPass.
         * @param run Finite idle pass or GPU-terminated captured interval.
         * @param device_id Exact GPU ordinal.
         * @param stream Non-null prepared worker stream.
         * @return Whether the native kernel was accepted; never changes mechanisms.
         */
        virtual bool enqueueMappedTransferService(
            const MappedTransferProgressCommand *commands,
            MappedTransferProgressCompletion *completions,
            MappedTransferServiceCursor *cursors, size_t capacity,
            size_t maximum_bytes, const std::uint32_t *interval,
            MappedTransferServiceRun run, int device_id, void *stream)
        {
            (void)commands; (void)completions; (void)cursors; (void)capacity;
            (void)maximum_bytes; (void)interval; (void)run; (void)device_id; (void)stream;
            return false;
        }

        // ====================================================================
        // Pinned Host Memory Allocation
        // ====================================================================

        /**
         * @brief Allocate pinned (page-locked) host memory for async DMA transfers
         *
         * Pinned memory enables true async H2D/D2H transfers without internal
         * staging copies. Required for overlapped pipeline transfers.
         *
         * @param bytes Number of bytes to allocate
         * @param device_id GPU device ID (for device affinity, 0-based)
         * @return Host pointer (nullptr on failure)
         *
         * **Semantics**:
         * - CUDA: cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault)
         * - ROCm: hipHostMalloc(&ptr, bytes, hipHostMallocDefault)
         * - CPU: malloc(bytes)
         *
         * **Lifetime**: Caller owns memory, must call freePinned()
         */
        virtual void *allocatePinned(size_t bytes, int device_id)
        {
            (void)bytes;
            (void)device_id;
            return nullptr;
        }

        /**
         * @brief Free pinned host memory allocated by allocatePinned()
         *
         * @param ptr Host pointer from allocatePinned() (may be nullptr)
         * @param device_id GPU device ID used for allocation
         */
        virtual void freePinned(void *ptr, int device_id)
        {
            (void)ptr;
            (void)device_id;
        }

        // ====================================================================
        // Stream-Aware Memory Operations
        // ====================================================================

        /**
         * @brief Async device-to-device copy on a specific stream
         *
         * Both src and dst must be device pointers on the same device (or
         * accessible from that device, e.g., BAR-mapped UVA pointers).
         *
         * @param dst Device destination pointer
         * @param src Device source pointer
         * @param bytes Number of bytes to copy
         * @param device_id GPU device ID (0-based)
         * @param stream Exact operation stream for GPU backends. GPU
         *               implementations reject nullptr.
         * @return true on success, false on error
         *
         * **Semantics**:
         * - CUDA: cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream)
         * - ROCm: hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream)
         * - CPU: memcpy(dst, src, bytes)
         */
        virtual bool deviceCopyAsync(void *dst, const void *src, size_t bytes,
                                     int device_id, void *stream)
        {
            (void)dst;
            (void)src;
            (void)bytes;
            (void)device_id;
            (void)stream;
            return false;
        }

        // ====================================================================
        // Collective Reduction Primitives
        // ====================================================================

        /**
         * @brief In-place element-wise vector addition: output += input
         *
         * Performs output[i] += input[i] for count elements on the GPU.
         * Used by cross-vendor collective backends for allreduce operations.
         *
         * @param output Device pointer to accumulate into (read+write)
         * @param input Device pointer to add (read-only)
         * @param count Number of elements
         * @param element_size Size of each element in bytes (4=FP32, 2=FP16/BF16, 1=INT8)
         * @param device_id GPU device ID (0-based)
         * @param stream Exact operation stream for GPU backends. GPU
         *               implementations reject nullptr.
         * @return true on success, false if not supported or error
         *
         * The element_size parameter determines the data type:
         * - 4 bytes: FP32 addition
         * - 2 bytes: FP16 or BF16 addition (backend-specific)
         * - 1 byte: INT8 saturating addition
         *
         * **Semantics**:
         * - CUDA: Launches vectorAdd kernel
         * - ROCm: Launches HIP vectorAdd kernel (future)
         * - CPU: Scalar loop
         */
        virtual bool vectorAddInplace(void *output, const void *input, size_t count,
                                      int element_size, int device_id, void *stream)
        {
            (void)output;
            (void)input;
            (void)count;
            (void)element_size;
            (void)device_id;
            (void)stream;
            return false;
        }

        // ====================================================================
        // Backend Identity
        // ====================================================================

        /**
         * @brief Return the DeviceType this backend handles
         *
         * Used for defensive validation in cross-device scenarios (e.g., verifying
         * that a GPU stream matches the backend before calling recordEvent).
         */
        virtual DeviceType backendDeviceType() const = 0;
    };

} // namespace llaminar2
