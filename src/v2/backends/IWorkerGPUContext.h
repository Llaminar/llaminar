#pragma once

#include <functional>
#include <future>
#include <memory>
#include <string>

namespace llaminar2
{

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

    protected:
        IWorkerGPUContext() = default;
    };

} // namespace llaminar2
