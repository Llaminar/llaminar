/**
 * @file NvidiaDeviceContext.h
 * @brief CUDA GPU device context with dedicated worker thread
 *
 * This class implements the IWorkerGPUContext interface for NVIDIA GPUs.
 * It owns a dedicated worker thread that initializes the CUDA runtime context,
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
#include <array>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <future>
#include <unordered_map>

namespace llaminar2
{

    /**
     * @class NvidiaDeviceContext
     * @brief CUDA device context with dedicated worker thread ownership
     *
     * This class provides:
     * - Dedicated worker thread for all CUDA operations
     * - CUDA runtime context initialization
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
         * releases CUDA resources (stream and cuBLAS handles).
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

        /** @copydoc IWorkerGPUContext::ownsCurrentThread() */
        bool ownsCurrentThread() const noexcept override;

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
        void *getOrCreateAuxiliaryStream(const std::string &name, bool *created = nullptr) override;
        /** @copydoc IWorkerGPUContext::getOrCreateAuxiliaryStream(const std::string &, GPUAuxiliaryStreamSchedulingClass, bool *) */
        void *getOrCreateAuxiliaryStream(
            const std::string &name,
            GPUAuxiliaryStreamSchedulingClass scheduling_class,
            bool *created = nullptr) override;
        void resetAuxiliaryStreams() override;

        // =========================================================================
        // IWorkerGPUContext Interface - Event Access (worker-thread-only)
        // =========================================================================

        void *createEvent() override;
        void destroyEvent(void *event) override;
        void recordEvent(void *event, void *stream) override;
        bool recordEventChecked(void *event, void *stream) override;
        void waitEvent(void *event, void *stream) override;
        bool waitEventChecked(void *event, void *stream) override;
        bool queryEventChecked(void *event, bool &ready) override;
        /** @copydoc IWorkerGPUContext::queryStreamExecutionState */
        [[nodiscard]] GPUStreamExecutionState queryStreamExecutionState(
            void *stream,
            std::string_view boundary) override;
        void synchronizeEvent(void *event) override;
        bool synchronizeEventChecked(void *event) override;
        float eventElapsedTime(void *start, void *stop) override;

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
        bool synchronizeChecked() override;
        void synchronizeStream(void *stream) override;
        bool synchronizeStreamChecked(void *stream) override;
        bool insertStreamDependency(
            void *dependent_stream,
            void *dependency_stream) override;

        std::unique_ptr<IGPUGraphCapture> createGraphCapture() override;
        std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *stream) override;
        PointerValidationResult validatePointerDevice(const void *gpu_ptr, int expected_ordinal) override;
        PointerInspectionResult inspectPointer(const void *gpu_ptr) const override;
        bool debugSynchronize() override;

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

        cudaStream_t default_stream_ = nullptr;
        std::unordered_map<std::string, cudaStream_t> auxiliary_streams_;
        /** Immutable scheduling class paired with every named stream. */
        std::unordered_map<
            std::string,
            GPUAuxiliaryStreamSchedulingClass>
            auxiliary_stream_scheduling_classes_;
        std::mutex auxiliary_streams_mutex_;

        /**
         * @brief Number of independently leasable inter-stream handoff events.
         *
         * A context normally has only a handful of concurrent graph owners.
         * Thirty-two slots leave ample headroom while keeping acquisition
         * lock-free in ordinary execution. A caller only waits on its selected
         * slot if more than thirty-two host threads concurrently publish a
         * dependency for the same physical GPU.
         */
        static constexpr size_t kStreamDependencyEventCount = 32;

        /// Persistent events used only for non-timing stream-to-stream handoffs.
        std::array<cudaEvent_t, kStreamDependencyEventCount>
            stream_dependency_events_{};

        /// Per-event lease flags prevent concurrent record/wait pairs from aliasing.
        std::array<std::atomic_flag, kStreamDependencyEventCount>
            stream_dependency_event_in_use_{};

        /// Round-robin ticket spreads simultaneous publishers across the event pool.
        std::atomic<size_t> next_stream_dependency_event_{0};

        cublasHandle_t cublas_handle_ = nullptr;
        cublasLtHandle_t cublas_lt_handle_ = nullptr;
        void *nccl_comm_ = nullptr;

        // =========================================================================
        // Worker Thread
        // =========================================================================

        std::thread worker_thread_;
        /// Immutable identity published before the context reports initialized.
        std::thread::id worker_thread_id_{};
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
