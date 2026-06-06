#pragma once

#include "IGPUGraphCapture.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>

namespace llaminar2
{

    struct PointerValidationResult
    {
        bool valid = true;
        int actual_device = -1;
        std::string details;
    };

    struct PointerInspectionResult
    {
        bool known = false;
        bool active = false;
        int actual_device = -1;
        const void *base_ptr = nullptr;
        size_t size_bytes = 0;
        uint64_t sequence = 0;
        uint64_t thread_hash = 0;
        std::string details;
    };

    /**
     * @brief Abstract interface for GPU device contexts
     *
     * Each GPU device has exactly one context that owns:
     * - The GPU context (CUDA primary context / HIP context)
     * - A dedicated worker thread
     * - Library handles (cuBLAS, hipBLAS, etc.)
     * - Event and stream pools
     * - Collective communicators (when initialized)
     *
     * ## Thread Safety Model
     *
     * This interface follows a strict ownership model where all GPU state is owned
     * by a dedicated worker thread. This eliminates data races and context switching
     * overhead that can occur when multiple threads access the same GPU.
     *
     * **Thread-safe methods** (can be called from any thread):
     * - `deviceOrdinal()`, `deviceName()`, `isInitialized()`
     * - `submitAndWait()`, `submitAsync()`
     * - `synchronize()`
     * - `collectiveComm()` (read-only access)
     *
     * **Worker-thread-only methods** (must be called from within submitted work):
     * - `defaultStream()`, `createStream()`, `destroyStream()`
     * - `createEvent()`, `destroyEvent()`, `recordEvent()`, `waitEvent()`, `synchronizeEvent()`
     * - `blasHandle()`
     * - `setCollectiveComm()` (write access)
     *
     * ## Usage Pattern
     *
     * ```cpp
     * // Submit GPU work from any thread
     * context->submitAndWait([&] {
     *     // Inside here, we're on the worker thread
     *     void* stream = context->defaultStream();
     *     void* handle = context->blasHandle();
     *     // ... launch kernels, call cuBLAS/hipBLAS ...
     * });
     *
     * // Or asynchronously
     * auto future = context->submitAsync([&] {
     *     // GPU work here
     * });
     * // ... do other work ...
     * future.wait();
     * ```
     *
     * ## Platform Abstraction
     *
     * This interface uses `void*` for platform-specific types to keep the header
     * platform-agnostic. Implementations cast these to the appropriate types:
     * - `void*` stream → `cudaStream_t` (CUDA) or `hipStream_t` (ROCm)
     * - `void*` event → `cudaEvent_t` (CUDA) or `hipEvent_t` (ROCm)
     * - `void*` blasHandle → `cublasHandle_t` (CUDA) or `hipblasHandle_t` (ROCm)
     * - `void*` collectiveComm → `ncclComm_t` (NCCL) or `ncclComm_t` (RCCL)
     */
    class IWorkerGPUContext
    {
    public:
        virtual ~IWorkerGPUContext() = default;

        // Non-copyable, non-movable (thread ownership semantics)
        IWorkerGPUContext(const IWorkerGPUContext &) = delete;
        IWorkerGPUContext &operator=(const IWorkerGPUContext &) = delete;
        IWorkerGPUContext(IWorkerGPUContext &&) = delete;
        IWorkerGPUContext &operator=(IWorkerGPUContext &&) = delete;

        // =========================================================================
        // Device Info (thread-safe, can be called from any thread)
        // =========================================================================

        /**
         * @brief Get the device ordinal (GPU index)
         * @return Device ordinal (0, 1, 2, ...)
         * @thread_safety Thread-safe, can be called from any thread
         */
        virtual int deviceOrdinal() const = 0;

        /**
         * @brief Get the human-readable device name
         * @return Device name string (e.g., "NVIDIA A100", "AMD MI250X")
         * @thread_safety Thread-safe, can be called from any thread
         */
        virtual std::string deviceName() const = 0;

        /**
         * @brief Check if the context is initialized and ready for work
         * @return true if initialized and worker thread is running
         * @thread_safety Thread-safe, can be called from any thread
         */
        virtual bool isInitialized() const = 0;

        // =========================================================================
        // Work Submission (thread-safe, can be called from any thread)
        // =========================================================================

        /**
         * @brief Submit work and wait for completion (blocking)
         *
         * The work function is executed on the dedicated worker thread that owns
         * the GPU context. This method blocks until the work completes.
         *
         * @param work Function to execute on worker thread
         * @throws std::runtime_error if worker thread is not running
         * @thread_safety Thread-safe, can be called from any thread
         *
         * @note The work function has access to all worker-thread-only methods
         */
        virtual void submitAndWait(std::function<void()> work) = 0;

        /**
         * @brief Submit work without waiting (non-blocking)
         *
         * The work function is queued for execution on the dedicated worker thread.
         * Returns immediately with a future that can be used to wait for completion.
         *
         * @param work Function to execute on worker thread
         * @return Future that completes when work is done
         * @throws std::runtime_error if worker thread is not running
         * @thread_safety Thread-safe, can be called from any thread
         *
         * @note The work function has access to all worker-thread-only methods
         */
        virtual std::future<void> submitAsync(std::function<void()> work) = 0;

        // =========================================================================
        // Stream Access (worker-thread-only)
        // =========================================================================

        /**
         * @brief Get the default stream for this device
         * @return Platform-specific stream handle (cudaStream_t or hipStream_t)
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note The default stream is created during context initialization and
         *       persists for the lifetime of the context
         */
        virtual void *defaultStream() = 0;

        /**
         * @brief Create a new stream on this device
         * @return Platform-specific stream handle (cudaStream_t or hipStream_t)
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note Caller is responsible for destroying the stream with destroyStream()
         */
        virtual void *createStream() = 0;

        /**
         * @brief Destroy a previously created stream
         * @param stream Stream handle returned by createStream()
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @warning Do not destroy the default stream
         */
        virtual void destroyStream(void *stream) = 0;

        // =========================================================================
        // Event Access (worker-thread-only)
        // =========================================================================

        /**
         * @brief Create a new event on this device
         * @return Platform-specific event handle (cudaEvent_t or hipEvent_t)
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note Caller is responsible for destroying the event with destroyEvent()
         */
        virtual void *createEvent() = 0;

        /**
         * @brief Destroy a previously created event
         * @param event Event handle returned by createEvent()
         * @thread_safety Must be called from worker thread (within submitted work)
         */
        virtual void destroyEvent(void *event) = 0;

        /**
         * @brief Record an event on a stream
         * @param event Event handle to record
         * @param stream Stream to record on (nullptr = default stream)
         * @thread_safety Must be called from worker thread (within submitted work)
         */
        virtual void recordEvent(void *event, void *stream = nullptr) = 0;

        /**
         * @brief Make a stream wait for an event
         * @param event Event handle to wait for
         * @param stream Stream that should wait (nullptr = default stream)
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note This is a GPU-side wait, not a CPU-side wait
         */
        virtual void waitEvent(void *event, void *stream = nullptr) = 0;

        /**
         * @brief Synchronize the CPU with an event (blocking)
         * @param event Event handle to synchronize with
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note This blocks the worker thread until the event completes on the GPU
         */
        virtual void synchronizeEvent(void *event) = 0;

        /**
         * @brief Compute elapsed time in milliseconds between two recorded events
         * @param start Event recorded before the operation
         * @param stop Event recorded after the operation
         * @return Elapsed time in milliseconds, or -1.0f on error
         * @thread_safety Must be called from worker thread after both events have completed
         *
         * @note Both events must have been recorded on the same device.
         *       The stop event must have completed (synchronize first).
         */
        virtual float eventElapsedTime(void *start, void *stop) = 0;

        // =========================================================================
        // Library Handles (worker-thread-only)
        // =========================================================================

        /**
         * @brief Get the BLAS library handle for this device
         * @return Platform-specific handle (cublasHandle_t or hipblasHandle_t)
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note The handle is created during context initialization and persists
         *       for the lifetime of the context
         */
        virtual void *blasHandle() = 0;

        /**
         * @brief Get the cuBLASLt/hipBLASLt handle for this device (for fused GEMM operations)
         * @return Raw handle pointer (cublasLtHandle_t or hipblasLtHandle_t cast to void*)
         *         nullptr if not available
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note The handle is created during context initialization and persists
         *       for the lifetime of the context. Used for fused operations like GEMM+bias.
         */
        virtual void *blasLtHandle() = 0;

        // =========================================================================
        // Collective Communicator (set by collective backend during initialization)
        // =========================================================================

        /**
         * @brief Set the collective communicator for this device
         * @param comm Platform-specific communicator (ncclComm_t for NCCL/RCCL)
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note Called by the collective backend (NCCL/RCCL) during initialization
         */
        virtual void setCollectiveComm(void *comm) = 0;

        /**
         * @brief Get the collective communicator for this device
         * @return Platform-specific communicator, or nullptr if not initialized
         * @thread_safety Thread-safe for read access
         */
        virtual void *collectiveComm() const = 0;

        // =========================================================================
        // Synchronization (thread-safe)
        // =========================================================================

        /**
         * @brief Wait for all work on this device to complete
         *
         * Submits a synchronization command to the worker thread and waits for
         * all pending GPU work to complete.
         *
         * @thread_safety Thread-safe, can be called from any thread
         */
        virtual void synchronize() = 0;

        /**
         * @brief Synchronize a specific stream (CPU blocks until stream completes)
         * @param stream Stream handle to synchronize. nullptr = legacy default stream (stream 0).
         * @thread_safety Thread-safe, can be called from any thread
         *
         * @note This is significantly cheaper than synchronize() (device-wide sync)
         *       because it only waits for work on one stream, not all streams.
         *       Especially important for segmented graph capture where alternating
         *       between graph launches (capture_stream) and manual stages (stream 0)
         *       requires ~50 sync points per decode token.
         */
        virtual void synchronizeStream(void *stream) = 0;

        /**
         * @brief Synchronize a stream and report whether the synchronization succeeded.
         *
         * This is the status-bearing companion to synchronizeStream(). Graph replay
         * paths should prefer this form so a failed CUDA/HIP stream sync stops
         * execution at the segment that poisoned the context instead of allowing
         * a later sampler or transfer to surface the stale error.
         *
         * @param stream Stream handle to synchronize. nullptr = legacy default stream (stream 0).
         * @return true when the stream completed successfully, false when the backend reported an error.
         * @thread_safety Thread-safe, can be called from any thread
         */
        virtual bool synchronizeStreamChecked(void *stream)
        {
            synchronizeStream(stream);
            return true;
        }

        /**
         * @brief Insert a GPU-side dependency between two streams (non-blocking from CPU)
         * @param dependent_stream Stream that should wait (nullptr = legacy stream 0)
         * @param dependency_stream Stream to wait for (nullptr = legacy stream 0)
         *
         * Makes dependent_stream wait until all prior work on dependency_stream
         * completes, using an internal event. This is a GPU-side wait — the CPU
         * is NOT blocked. This is much cheaper than synchronizeStream() because
         * it avoids CPU stalls entirely.
         *
         * Used by segmented graph capture to order graph launches (on capture_stream)
         * with manual stage dispatches (on legacy stream 0) without CPU overhead.
         *
         * @note The event is managed internally; callers don't need to create/destroy events.
         */
        virtual void insertStreamDependency(void *dependent_stream, void *dependency_stream) = 0;

        // =========================================================================
        // GPU Graph Capture (worker-thread-only)
        // =========================================================================

        /**
         * @brief Create a GPU graph capture object for this device's default stream
         * @return Unique pointer to a backend-specific graph capture implementation
         * @thread_safety Must be called from worker thread (within submitted work)
         *
         * @note The returned object captures from this context's default stream.
         *       The graph capture object must not outlive this context.
         */
        virtual std::unique_ptr<IGPUGraphCapture> createGraphCapture() = 0;

        /**
         * @brief Create a GPU graph capture object using a specific stream
         * @param stream Opaque GPU stream pointer (hipStream_t or cudaStream_t cast to void*)
         * @return Unique pointer to a backend-specific graph capture implementation
         *
         * This overload is used for segmented graph capture, where the capture stream
         * is created locally rather than using the device context's default stream.
         */
        virtual std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *stream) = 0;

        // =========================================================================
        // Diagnostics and Debug Utilities
        // =========================================================================

        virtual void clearLastError() {}

        virtual PointerValidationResult validatePointerDevice(const void *gpu_ptr, int expected_ordinal)
        {
            (void)gpu_ptr;
            return {true, expected_ordinal, ""};
        }

        virtual PointerInspectionResult inspectPointer(const void *gpu_ptr) const
        {
            (void)gpu_ptr;
            return {};
        }

        virtual void dumpRecentPointerEvents(size_t max_events) { (void)max_events; }

        virtual bool debugSynchronize()
        {
            synchronize();
            return true;
        }

        // =========================================================================
        // Graph Capture State
        // =========================================================================

        /**
         * @brief Indicate that a graph capture recording is active on this device.
         *
         * While active, operations that are illegal during HIP/CUDA graph capture
         * (e.g., hipDeviceSynchronize, hipStreamSynchronize on the capture stream,
         * synchronous hipMemcpy) will be skipped or handled specially.
         *
         * @param active true when between beginCapture/endCapture, false otherwise
         * @thread_safety Thread-safe; uses atomic internally
         */
        virtual void setGraphCaptureActive(bool active) { (void)active; }

        /**
         * @brief Check whether graph capture recording is active on this device.
         * @return true if between beginCapture/endCapture
         */
        virtual bool isDeviceGraphCaptureActive() const { return false; }

    protected:
        IWorkerGPUContext() = default;
    };

} // namespace llaminar2
