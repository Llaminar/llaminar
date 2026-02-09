/**
 * @file AMDDeviceContext.h
 * @brief ROCm/HIP GPU device context with dedicated worker thread
 *
 * This class implements the IWorkerGPUContext interface for AMD GPUs.
 * It owns a dedicated worker thread that holds the HIP context,
 * ensuring all HIP operations are executed from a single thread to avoid
 * context switching overhead and thread-safety issues.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../IWorkerGPUContext.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <future>

namespace llaminar2
{

    /**
     * @class AMDDeviceContext
     * @brief ROCm/HIP device context with dedicated worker thread ownership
     *
     * This class provides:
     * - Dedicated worker thread for all HIP operations
     * - HIP context initialization and management
     * - hipBLAS handle creation and management
     * - Stream and event creation/destruction
     * - Thread-safe work submission via submitAndWait() / submitAsync()
     *
     * ## Usage Example
     *
     * ```cpp
     * // Create context for GPU 0
     * auto ctx = std::make_unique<AMDDeviceContext>(0);
     *
     * // Submit GPU work
     * ctx->submitAndWait([&] {
     *     hipStream_t stream = static_cast<hipStream_t>(ctx->defaultStream());
     *     hipblasHandle_t handle = static_cast<hipblasHandle_t>(ctx->blasHandle());
     *     // ... launch kernels, call hipBLAS ...
     * });
     * ```
     *
     * ## Thread Safety
     *
     * - submitAndWait(), submitAsync(), synchronize() are thread-safe
     * - Stream/event/handle accessors must only be called from within submitted work
     * - deviceOrdinal(), deviceName(), isInitialized() are thread-safe (read-only)
     */
    class AMDDeviceContext : public IWorkerGPUContext
    {
    public:
        /**
         * @brief Construct a ROCm/HIP device context
         * @param device_ordinal GPU ordinal (0, 1, 2, ...)
         * @throws std::runtime_error if worker thread fails to initialize
         *
         * The constructor starts the worker thread and blocks until the HIP
         * context is fully initialized on the worker thread.
         */
        explicit AMDDeviceContext(int device_ordinal);

        /**
         * @brief Destructor - shuts down worker thread and releases resources
         *
         * Signals the worker thread to exit, waits for it to complete, and
         * releases all HIP resources (stream, hipBLAS handle).
         */
        ~AMDDeviceContext() override;

        // =========================================================================
        // IWorkerGPUContext Interface - Device Info (thread-safe)
        // =========================================================================

        int deviceOrdinal() const override { return device_ordinal_; }
        std::string deviceName() const override { return device_name_; }
        bool isInitialized() const override { return initialized_.load(); }

        // =========================================================================
        // IWorkerGPUContext Interface - Work Submission (thread-safe)
        // =========================================================================

        /**
         * @brief Submit work and wait for completion (blocking)
         * @param work Function to execute on worker thread
         */
        void submitAndWait(std::function<void()> work) override;

        /**
         * @brief Submit work without waiting (non-blocking)
         * @param work Function to execute on worker thread
         * @return Future that completes when work is done
         */
        std::future<void> submitAsync(std::function<void()> work) override;

        // =========================================================================
        // IWorkerGPUContext Interface - Stream Access (worker-thread-only)
        // =========================================================================

        void *defaultStream() override;
        void *createStream() override;
        void destroyStream(void *stream) override;

        // =========================================================================
        // IWorkerGPUContext Interface - Event Access (worker-thread-only)
        // =========================================================================

        void *createEvent() override;
        void destroyEvent(void *event) override;
        void recordEvent(void *event, void *stream) override;
        void waitEvent(void *event, void *stream) override;
        void synchronizeEvent(void *event) override;

        // =========================================================================
        // IWorkerGPUContext Interface - Library Handles (worker-thread-only)
        // =========================================================================

        void *blasHandle() override;
        void *blasLtHandle() override;

        // =========================================================================
        // IWorkerGPUContext Interface - Collective Communicator
        // =========================================================================

        void setCollectiveComm(void *comm) override;
        void *collectiveComm() const override;

        // =========================================================================
        // IWorkerGPUContext Interface - Synchronization (thread-safe)
        // =========================================================================

        void synchronize() override;
        void synchronizeStream(void *stream) override;
        void insertStreamDependency(void *dependent_stream, void *dependency_stream) override;

        std::unique_ptr<IGPUGraphCapture> createGraphCapture() override;
        std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *stream) override;

    private:
        // =========================================================================
        // Worker Thread Management
        // =========================================================================

        /**
         * @brief Main worker thread loop
         *
         * Initializes HIP context on entry, processes work queue until shutdown,
         * then cleans up resources on exit.
         */
        void workerLoop();

        /**
         * @brief Initialize HIP resources on worker thread
         * @return true if initialization succeeded
         *
         * Called from workerLoop() to:
         * - Set HIP device via hipSetDevice()
         * - Create default stream
         * - Create hipBLAS handle
         * - Query device name
         */
        bool initializeOnWorker();

        /**
         * @brief Cleanup HIP resources on worker thread
         *
         * Called from workerLoop() before exit to:
         * - Destroy hipBLAS handle
         * - Destroy default stream
         */
        void cleanupOnWorker();

        // =========================================================================
        // Device Info
        // =========================================================================

        int device_ordinal_;
        std::string device_name_;
        std::atomic<bool> initialized_{false};

        // =========================================================================
        // GPU State (owned by worker thread)
        // =========================================================================

        hipStream_t default_stream_ = nullptr;
        hipblasHandle_t hipblas_handle_ = nullptr;
        hipblasLtHandle_t hipblas_lt_handle_ = nullptr;
        void *rccl_comm_ = nullptr;

        // =========================================================================
        // Worker Thread
        // =========================================================================

        std::thread worker_thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> shutdown_requested_{false};

        // =========================================================================
        // Work Queue
        // =========================================================================

        std::queue<std::packaged_task<void()>> work_queue_;
        std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
    };

} // namespace llaminar2
