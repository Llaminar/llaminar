/**
 * @file NvidiaDeviceContext.h
 * @brief CUDA GPU device context with dedicated worker thread
 *
 * This class implements the IWorkerGPUContext interface for NVIDIA GPUs.
 * It owns a dedicated worker thread that holds the CUDA primary context,
 * ensuring all CUDA operations are executed from a single thread to avoid
 * context switching overhead and thread-safety issues.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../IWorkerGPUContext.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
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
     * @class NvidiaDeviceContext
     * @brief CUDA device context with dedicated worker thread ownership
     *
     * This class provides:
     * - Dedicated worker thread for all CUDA operations
     * - CUDA primary context retention via driver API
     * - cuBLAS handle creation and management
     * - Stream and event creation/destruction
     * - Thread-safe work submission via submitAndWait() / submitAsync()
     *
     * ## Usage Example
     *
     * ```cpp
     * // Create context for GPU 0
     * auto ctx = std::make_unique<NvidiaDeviceContext>(0);
     *
     * // Submit GPU work
     * ctx->submitAndWait([&] {
     *     cudaStream_t stream = static_cast<cudaStream_t>(ctx->defaultStream());
     *     cublasHandle_t handle = static_cast<cublasHandle_t>(ctx->blasHandle());
     *     // ... launch kernels, call cuBLAS ...
     * });
     * ```
     *
     * ## Thread Safety
     *
     * - submitAndWait(), submitAsync(), synchronize() are thread-safe
     * - Stream/event/handle accessors must only be called from within submitted work
     * - deviceOrdinal(), deviceName(), isInitialized() are thread-safe (read-only)
     */
    class NvidiaDeviceContext : public IWorkerGPUContext
    {
    public:
        /**
         * @brief Construct a CUDA device context
         * @param device_ordinal GPU ordinal (0, 1, 2, ...)
         * @throws std::runtime_error if worker thread fails to initialize
         *
         * The constructor starts the worker thread and blocks until the CUDA
         * context is fully initialized on the worker thread.
         */
        explicit NvidiaDeviceContext(int device_ordinal);

        /**
         * @brief Destructor - shuts down worker thread and releases resources
         *
         * Signals the worker thread to exit, waits for it to complete, and
         * releases all CUDA resources (stream, cuBLAS handle, context).
         */
        ~NvidiaDeviceContext() override;

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
         * Initializes CUDA context on entry, processes work queue until shutdown,
         * then cleans up resources on exit.
         */
        void workerLoop();

        /**
         * @brief Initialize CUDA resources on worker thread
         * @return true if initialization succeeded
         *
         * Called from workerLoop() to:
         * - Set CUDA device via cudaSetDevice()
         * - Retain primary context via cuDevicePrimaryCtxRetain()
         * - Create default stream
         * - Create cuBLAS handle
         * - Query device name
         */
        bool initializeOnWorker();

        /**
         * @brief Cleanup CUDA resources on worker thread
         *
         * Called from workerLoop() before exit to:
         * - Destroy cuBLAS handle
         * - Destroy default stream
         * - Release primary context (if retained)
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

        cudaStream_t default_stream_ = nullptr;
        cublasHandle_t cublas_handle_ = nullptr;
        cublasLtHandle_t cublas_lt_handle_ = nullptr;
        void *cuda_context_ = nullptr; // CUcontext from driver API
        void *nccl_comm_ = nullptr;

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
