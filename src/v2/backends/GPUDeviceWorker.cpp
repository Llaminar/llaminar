/**
 * @file GPUDeviceWorker.cpp
 * @brief Implementation of GPU device worker threads (platform-independent parts)
 *
 * This file contains the thread management logic that is independent of
 * CUDA vs ROCm. The GPU-specific initialization/cleanup is delegated to
 * platform-specific helper functions that are implemented in the
 * respective backend files.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "GPUDeviceWorker.h"

namespace llaminar2
{

    // =========================================================================
    // External GPU platform functions (implemented in respective backend files)
    // =========================================================================

    // Forward declarations - these are implemented in CUDABackend.cu and ROCmBackend.cpp
    // to avoid including conflicting headers in this file.

    // CUDA implementations (defined in CUDABackend.cu)
    namespace cuda_worker
    {
        bool initializeContext(int ordinal, void **out_stream, void **out_context);
        void cleanupContext(int ordinal, void *stream, void *context);
    }

    // ROCm implementations (defined in ROCmBackend.cpp)
    namespace rocm_worker
    {
        bool initializeContext(int ordinal, void **out_stream, void **out_context);
        void cleanupContext(int ordinal, void *stream, void *context);
    }

    // =========================================================================
    // GPUDeviceWorker Implementation
    // =========================================================================

    GPUDeviceWorker::GPUDeviceWorker(DeviceId device)
        : device_(device)
    {
    }

    GPUDeviceWorker::~GPUDeviceWorker()
    {
        stop();
    }

    bool GPUDeviceWorker::start()
    {
        if (running_.load())
        {
            return true; // Already running
        }

        // Validate device type
        if (!device_.is_cuda() && !device_.is_rocm())
        {
            LOG_ERROR("[GPUDeviceWorker] Cannot create worker for non-GPU device: " << device_.toString());
            return false;
        }

        stop_requested_.store(false);

        try
        {
            thread_ = std::thread(&GPUDeviceWorker::workerLoop, this);

            // Wait for worker to signal it's ready
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]()
                     { return running_.load() || stop_requested_.load(); });

            if (running_.load())
            {
                LOG_DEBUG("[GPUDeviceWorker] Started worker for device " << device_.toString());
                return true;
            }
            else
            {
                // Worker failed to initialize
                if (thread_.joinable())
                {
                    thread_.join();
                }
                return false;
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[GPUDeviceWorker] Failed to start worker thread: " << e.what());
            return false;
        }
    }

    void GPUDeviceWorker::stop()
    {
        if (!running_.load() && !thread_.joinable())
        {
            return;
        }

        // Signal worker to stop
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_requested_.store(true);
        }
        cv_.notify_one();

        // Wait for worker to finish
        if (thread_.joinable())
        {
            thread_.join();
        }

        running_.store(false);
        LOG_DEBUG("[GPUDeviceWorker] Stopped worker for device " << device_.toString());
    }

    std::future<bool> GPUDeviceWorker::submit(GPUWorkItem work)
    {
        // Wrap work to capture stream
        auto wrapper = [this, w = std::move(work)](void *) -> bool
        {
            return w(stream_);
        };

        std::packaged_task<bool(void *)> task(std::move(wrapper));
        std::future<bool> future = task.get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_.load())
            {
                LOG_ERROR("[GPUDeviceWorker] Cannot submit work - worker not running");
                // Return a ready future with false
                std::promise<bool> p;
                p.set_value(false);
                return p.get_future();
            }
            work_queue_.push(std::move(task));
        }
        cv_.notify_one();

        return future;
    }

    bool GPUDeviceWorker::submitAndWait(GPUWorkItem work)
    {
        auto future = submit(std::move(work));
        return future.get();
    }

    void GPUDeviceWorker::workerLoop()
    {
        // Initialize context on this thread
        if (!initializeContext())
        {
            LOG_ERROR("[GPUDeviceWorker] Failed to initialize context for " << device_.toString());
            stop_requested_.store(true); // Signal failure
            cv_.notify_all();
            return;
        }

        // Signal ready
        running_.store(true);
        cv_.notify_all();

        LOG_TRACE("[GPUDeviceWorker] Worker loop started for " << device_.toString());

        // Work loop
        while (true)
        {
            std::packaged_task<bool(void *)> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]()
                         { return stop_requested_.load() || !work_queue_.empty(); });

                if (stop_requested_.load() && work_queue_.empty())
                {
                    break;
                }

                if (!work_queue_.empty())
                {
                    task = std::move(work_queue_.front());
                    work_queue_.pop();
                }
            }

            if (task.valid())
            {
                // Execute the task (context is set on this thread)
                task(stream_);
            }
        }

        // Cleanup
        cleanupContext();
        running_.store(false);
        LOG_TRACE("[GPUDeviceWorker] Worker loop exited for " << device_.toString());
    }

    bool GPUDeviceWorker::initializeContext()
    {
        int ordinal = device_.gpu_ordinal();

        if (device_.is_cuda())
        {
#ifdef HAVE_CUDA
            return cuda_worker::initializeContext(ordinal, &stream_, &context_);
#else
            LOG_ERROR("[GPUDeviceWorker] CUDA not compiled in");
            return false;
#endif
        }
        else if (device_.is_rocm())
        {
#ifdef HAVE_ROCM
            return rocm_worker::initializeContext(ordinal, &stream_, &context_);
#else
            LOG_ERROR("[GPUDeviceWorker] ROCm not compiled in");
            return false;
#endif
        }

        LOG_ERROR("[GPUDeviceWorker] Unknown device type: " << device_.toString());
        return false;
    }

    void GPUDeviceWorker::cleanupContext()
    {
        int ordinal = device_.gpu_ordinal();

        if (device_.is_cuda())
        {
#ifdef HAVE_CUDA
            cuda_worker::cleanupContext(ordinal, stream_, context_);
#endif
        }
        else if (device_.is_rocm())
        {
#ifdef HAVE_ROCM
            rocm_worker::cleanupContext(ordinal, stream_, context_);
#endif
        }

        stream_ = nullptr;
        context_ = nullptr;
    }

    // =========================================================================
    // GPUDeviceWorkerPool Implementation
    // =========================================================================

    GPUDeviceWorkerPool &GPUDeviceWorkerPool::instance()
    {
        static GPUDeviceWorkerPool pool;
        return pool;
    }

    GPUDeviceWorkerPool::~GPUDeviceWorkerPool()
    {
        shutdown();
    }

    bool GPUDeviceWorkerPool::initialize(const std::vector<DeviceId> &devices)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if already initialized with same devices
        if (initialized_ && hasWorkersUnlocked(devices))
        {
            return true;
        }

        // Create workers for any new devices
        for (const auto &device : devices)
        {
            // Check if we already have this worker
            bool found = false;
            for (const auto &worker : workers_)
            {
                if (worker->device() == device)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                auto worker = std::make_unique<GPUDeviceWorker>(device);
                if (!worker->start())
                {
                    LOG_ERROR("[GPUDeviceWorkerPool] Failed to start worker for " << device.toString());
                    return false;
                }
                workers_.push_back(std::move(worker));
            }
        }

        initialized_ = true;
        LOG_DEBUG("[GPUDeviceWorkerPool] Initialized with " << workers_.size() << " workers");
        return true;
    }

    void GPUDeviceWorkerPool::shutdown()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &worker : workers_)
        {
            worker->stop();
        }
        workers_.clear();
        initialized_ = false;

        LOG_DEBUG("[GPUDeviceWorkerPool] Shutdown complete");
    }

    GPUDeviceWorker *GPUDeviceWorkerPool::getWorker(const DeviceId &device)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &worker : workers_)
        {
            if (worker->device() == device)
            {
                return worker.get();
            }
        }
        return nullptr;
    }

    std::future<bool> GPUDeviceWorkerPool::submit(const DeviceId &device, GPUWorkItem work)
    {
        auto *worker = getWorker(device);
        if (!worker)
        {
            LOG_WARN("[GPUDeviceWorkerPool] No worker for device " << device.toString());
            std::promise<bool> p;
            p.set_value(false);
            return p.get_future();
        }
        return worker->submit(std::move(work));
    }

    bool GPUDeviceWorkerPool::submitAndWait(const DeviceId &device, GPUWorkItem work)
    {
        auto *worker = getWorker(device);
        if (!worker)
        {
            LOG_WARN("[GPUDeviceWorkerPool] No worker for device " << device.toString());
            return false;
        }
        return worker->submitAndWait(std::move(work));
    }

    bool GPUDeviceWorkerPool::hasWorkers(const std::vector<DeviceId> &devices) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return hasWorkersUnlocked(devices);
    }

    bool GPUDeviceWorkerPool::hasWorkersUnlocked(const std::vector<DeviceId> &devices) const
    {
        for (const auto &device : devices)
        {
            bool found = false;
            for (const auto &worker : workers_)
            {
                if (worker->device() == device && worker->isRunning())
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                return false;
            }
        }
        return true;
    }

    std::vector<DeviceId> GPUDeviceWorkerPool::managedDevices() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<DeviceId> devices;
        devices.reserve(workers_.size());
        for (const auto &worker : workers_)
        {
            devices.push_back(worker->device());
        }
        return devices;
    }

} // namespace llaminar2
