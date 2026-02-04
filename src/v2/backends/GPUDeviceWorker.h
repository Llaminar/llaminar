/**
 * @file GPUDeviceWorker.h
 * @brief Dedicated worker thread per GPU device for context-safe operations
 *
 * Provides a dedicated thread per GPU device that maintains a stable CUDA/HIP
 * context. This is critical for multi-GPU scenarios where collective library
 * initialization (NCCL/RCCL) can corrupt CUDA/HIP contexts on other threads.
 *
 * The worker thread:
 * 1. Initializes and retains the primary context for its assigned device
 * 2. Creates a dedicated stream for operations
 * 3. Processes work items submitted via a thread-safe queue
 * 4. Ensures all GPU operations execute in a clean context
 *
 * Usage:
 *   GPUDeviceWorker worker(DeviceId::cuda(0));
 *   worker.start();
 *
 *   // Submit work to execute on the worker's context
 *   auto future = worker.submit([](void* stream) {
 *       // GPU operations here - context is guaranteed stable
 *       return true;
 *   });
 *   bool result = future.get();
 *
 *   worker.stop();
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "DeviceId.h"
#include "../utils/Logger.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Work item type: function taking stream pointer, returning bool
     */
    using GPUWorkItem = std::function<bool(void *stream)>;

    /**
     * @brief Dedicated worker thread for a single GPU device
     *
     * Maintains a stable CUDA/HIP context that won't be corrupted by
     * collective library initialization on other threads.
     */
    class GPUDeviceWorker
    {
    public:
        /**
         * @brief Construct a worker for a specific device
         * @param device The GPU device this worker manages
         */
        explicit GPUDeviceWorker(DeviceId device);

        /**
         * @brief Destructor - stops worker if running
         */
        ~GPUDeviceWorker();

        // Non-copyable, non-movable (owns thread)
        GPUDeviceWorker(const GPUDeviceWorker &) = delete;
        GPUDeviceWorker &operator=(const GPUDeviceWorker &) = delete;
        GPUDeviceWorker(GPUDeviceWorker &&) = delete;
        GPUDeviceWorker &operator=(GPUDeviceWorker &&) = delete;

        /**
         * @brief Start the worker thread
         * @return true if started successfully (or already running)
         */
        bool start();

        /**
         * @brief Stop the worker thread
         *
         * Waits for pending work to complete, then joins the thread.
         */
        void stop();

        /**
         * @brief Check if worker is running
         */
        bool isRunning() const { return running_.load(); }

        /**
         * @brief Submit work to execute on this device's context
         *
         * @param work Function to execute - receives stream pointer
         * @return Future for the result
         */
        std::future<bool> submit(GPUWorkItem work);

        /**
         * @brief Submit work and wait for completion
         *
         * Convenience method that submits and blocks until done.
         *
         * @param work Function to execute
         * @return Result of the work function
         */
        bool submitAndWait(GPUWorkItem work);

        /**
         * @brief Get the device this worker manages
         */
        const DeviceId &device() const { return device_; }

        /**
         * @brief Get the worker's stream (only valid while running)
         *
         * @warning Only use this from within a submitted work item!
         */
        void *stream() const { return stream_; }

    private:
        /**
         * @brief Worker thread main loop
         */
        void workerLoop();

        /**
         * @brief Initialize GPU context on worker thread
         * @return true on success
         */
        bool initializeContext();

        /**
         * @brief Cleanup GPU resources on worker thread
         */
        void cleanupContext();

        DeviceId device_;
        std::thread thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> stop_requested_{false};

        // Work queue
        std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<std::packaged_task<bool(void *)>> work_queue_;

        // GPU resources (owned by worker thread)
        void *stream_ = nullptr;
        void *context_ = nullptr; // CUcontext or hipCtx_t (driver API handle)
    };

    /**
     * @brief Manager for multiple GPU device workers
     *
     * Provides a pool of workers, one per device, with convenient
     * access patterns.
     */
    class GPUDeviceWorkerPool
    {
    public:
        /**
         * @brief Get the singleton instance
         */
        static GPUDeviceWorkerPool &instance();

        /**
         * @brief Initialize workers for a set of devices
         *
         * @param devices List of devices to create workers for
         * @return true if all workers started successfully
         */
        bool initialize(const std::vector<DeviceId> &devices);

        /**
         * @brief Shutdown all workers
         */
        void shutdown();

        /**
         * @brief Get worker for a specific device
         *
         * @param device The device
         * @return Pointer to worker, or nullptr if not found
         */
        GPUDeviceWorker *getWorker(const DeviceId &device);

        /**
         * @brief Submit work to a specific device's worker
         *
         * @param device Target device
         * @param work Work to execute
         * @return Future for result, or invalid future if worker not found
         */
        std::future<bool> submit(const DeviceId &device, GPUWorkItem work);

        /**
         * @brief Submit work and wait for completion
         *
         * @param device Target device
         * @param work Work to execute
         * @return Result, or false if worker not found
         */
        bool submitAndWait(const DeviceId &device, GPUWorkItem work);

        /**
         * @brief Check if workers are initialized for given devices
         */
        bool hasWorkers(const std::vector<DeviceId> &devices) const;

        /**
         * @brief Get all managed devices
         */
        std::vector<DeviceId> managedDevices() const;

    private:
        GPUDeviceWorkerPool() = default;
        ~GPUDeviceWorkerPool();

        // Internal helper (must be called with mutex held)
        bool hasWorkersUnlocked(const std::vector<DeviceId> &devices) const;

        mutable std::mutex mutex_;
        std::vector<std::unique_ptr<GPUDeviceWorker>> workers_;
        bool initialized_ = false;
    };

    // =========================================================================
    // RAII Helper for scoped worker pool initialization
    // =========================================================================

    /**
     * @brief RAII helper to ensure worker pool is initialized for devices
     *
     * Usage:
     *   {
     *       GPUWorkerPoolScope scope({DeviceId::cuda(0), DeviceId::cuda(1)});
     *       // Workers are guaranteed running in this scope
     *   }
     *   // Workers still running (pool is a singleton)
     */
    class GPUWorkerPoolScope
    {
    public:
        explicit GPUWorkerPoolScope(const std::vector<DeviceId> &devices)
        {
            GPUDeviceWorkerPool::instance().initialize(devices);
        }
        // Note: We don't shutdown on destruction - pool is long-lived
    };

} // namespace llaminar2
