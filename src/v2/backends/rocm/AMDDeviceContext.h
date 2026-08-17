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
        void dumpRecentPointerEvents(size_t max_events) override;
        bool debugSynchronize() override;

        void setGraphCaptureActive(bool active) override { capture_active_.store(active, std::memory_order_release); }
        bool isDeviceGraphCaptureActive() const override { return capture_active_.load(std::memory_order_acquire); }

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
        std::unordered_map<std::string, hipStream_t> auxiliary_streams_;
        std::mutex auxiliary_streams_mutex_;

        /**
         * @brief Number of independently leasable inter-stream handoff events.
         *
         * The fixed pool removes HIP event allocation from graph replay and
         * gives concurrent graph owners separate record/wait pairs. Contention
         * remains local to one slot only after more than thirty-two concurrent
         * publishers target the same physical device.
         */
        static constexpr size_t kStreamDependencyEventCount = 32;

        /// Persistent events used only for non-timing stream-to-stream handoffs.
        std::array<hipEvent_t, kStreamDependencyEventCount>
            stream_dependency_events_{};

        /// Per-event lease flags prevent concurrent record/wait pairs from aliasing.
        std::array<std::atomic_flag, kStreamDependencyEventCount>
            stream_dependency_event_in_use_{};

        /// Round-robin ticket spreads simultaneous publishers across the event pool.
        std::atomic<size_t> next_stream_dependency_event_{0};

        hipblasHandle_t hipblas_handle_ = nullptr;
        hipblasLtHandle_t hipblas_lt_handle_ = nullptr;
        void *rccl_comm_ = nullptr;

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

        // =========================================================================
        // Graph Capture State
        // =========================================================================

        /// True while a HIP graph capture recording is active on this device.
        /// During capture, hipDeviceSynchronize() and other sync calls are illegal.
        std::atomic<bool> capture_active_{false};
    };

} // namespace llaminar2
