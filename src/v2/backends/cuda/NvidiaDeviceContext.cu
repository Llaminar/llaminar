/**
 * @file NvidiaDeviceContext.cu
 * @brief CUDA device context implementation with dedicated worker thread
 *
 * This file implements the NvidiaDeviceContext class, which provides:
 * - Dedicated worker thread for all CUDA operations
 * - CUDA runtime context initialization
 * - cuBLAS handle creation and management
 * - Stream and event creation/destruction
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "NvidiaDeviceContext.h"
#include "CUDAGraphCapture.h"
#include "../../utils/Logger.h"
#include <sstream>
#include <stdexcept>

namespace llaminar2
{

    // ============================================================================
    // Helper Macros for CUDA Error Checking
    // ============================================================================

#define CUDA_CHECK(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            LOG_ERROR("[NvidiaDeviceContext] " << #call << " failed: "      \
                                               << cudaGetErrorString(err)); \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_VOID(call)                                               \
    do                                                                      \
    {                                                                       \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            LOG_ERROR("[NvidiaDeviceContext] " << #call << " failed: "      \
                                               << cudaGetErrorString(err)); \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                             \
    do                                                                 \
    {                                                                  \
        cublasStatus_t status = (call);                                \
        if (status != CUBLAS_STATUS_SUCCESS)                           \
        {                                                              \
            LOG_ERROR("[NvidiaDeviceContext] " << #call << " failed: " \
                                               << status);             \
            return false;                                              \
        }                                                              \
    } while (0)

    // ============================================================================
    // Constructor / Destructor
    // ============================================================================

    NvidiaDeviceContext::NvidiaDeviceContext(int device_ordinal)
        : device_ordinal_(device_ordinal)
    {
        LOG_DEBUG("[NvidiaDeviceContext] Creating context for CUDA device " << device_ordinal);

        // Validate device ordinal before starting worker
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "cudaGetDeviceCount failed: " << cudaGetErrorString(err);
            throw std::runtime_error(oss.str());
        }

        if (device_ordinal < 0 || device_ordinal >= device_count)
        {
            std::ostringstream oss;
            oss << "Invalid CUDA device ordinal " << device_ordinal
                << " (available: 0-" << (device_count - 1) << ")";
            throw std::runtime_error(oss.str());
        }

        // Start worker thread
        shutdown_requested_.store(false);
        running_.store(false);

        try
        {
            worker_thread_ = std::thread(&NvidiaDeviceContext::workerLoop, this);

            // Wait for worker to signal it's ready (or failed)
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]()
                           { return initialized_.load() || shutdown_requested_.load(); });

            if (!initialized_.load())
            {
                // Worker failed to initialize - join thread and throw
                if (worker_thread_.joinable())
                {
                    worker_thread_.join();
                }
                throw std::runtime_error("Failed to initialize CUDA context on worker thread");
            }

            LOG_INFO("[NvidiaDeviceContext] Created context for CUDA device " << device_ordinal
                                                                              << " (" << device_name_ << ")");
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[NvidiaDeviceContext] Failed to start worker thread: " << e.what());
            throw;
        }
    }

    NvidiaDeviceContext::~NvidiaDeviceContext()
    {
        LOG_DEBUG("[NvidiaDeviceContext] Destroying context for CUDA device " << device_ordinal_);

        // Signal worker to shutdown
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            shutdown_requested_.store(true);
        }
        queue_cv_.notify_one();

        // Wait for worker to finish
        if (worker_thread_.joinable())
        {
            worker_thread_.join();
        }

        LOG_DEBUG("[NvidiaDeviceContext] Context destroyed for CUDA device " << device_ordinal_);
    }

    // ============================================================================
    // Worker Thread Implementation
    // ============================================================================

    void NvidiaDeviceContext::workerLoop()
    {
        LOG_TRACE("[NvidiaDeviceContext] Worker thread starting for device " << device_ordinal_);

        // Initialize CUDA context on this thread
        if (!initializeOnWorker())
        {
            LOG_ERROR("[NvidiaDeviceContext] Failed to initialize CUDA context on worker thread");
            shutdown_requested_.store(true);
            queue_cv_.notify_all();
            return;
        }

        // Signal that initialization is complete
        initialized_.store(true);
        running_.store(true);
        queue_cv_.notify_all();

        LOG_TRACE("[NvidiaDeviceContext] Worker loop started for device " << device_ordinal_);

        // Main work loop
        while (true)
        {
            std::packaged_task<void()> task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // Wait for work or shutdown signal
                queue_cv_.wait(lock, [this]()
                               { return shutdown_requested_.load() || !work_queue_.empty(); });

                // Check for shutdown (drain queue first)
                if (shutdown_requested_.load() && work_queue_.empty())
                {
                    break;
                }

                // Dequeue work
                if (!work_queue_.empty())
                {
                    task = std::move(work_queue_.front());
                    work_queue_.pop();
                }
            }

            // Execute task outside of lock
            if (task.valid())
            {
                try
                {
                    task();
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[NvidiaDeviceContext] Exception in worker task: " << e.what());
                }
                catch (...)
                {
                    LOG_ERROR("[NvidiaDeviceContext] Unknown exception in worker task");
                }
            }
        }

        // Cleanup before exiting
        cleanupOnWorker();
        running_.store(false);

        LOG_TRACE("[NvidiaDeviceContext] Worker loop exited for device " << device_ordinal_);
    }

    bool NvidiaDeviceContext::initializeOnWorker()
    {
        LOG_TRACE("[NvidiaDeviceContext] Initializing CUDA on worker thread for device "
                  << device_ordinal_);

        // Step 1: Set CUDA device via runtime API
        CUDA_CHECK(cudaSetDevice(device_ordinal_));

        // Step 2: Create default stream (non-blocking for better concurrency)
        CUDA_CHECK(cudaStreamCreateWithFlags(&default_stream_, cudaStreamNonBlocking));

        // Step 3: Create cuBLAS handle
        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[NvidiaDeviceContext] cublasCreate failed: " << cublas_status);
            return false;
        }

        // Associate cuBLAS handle with our stream
        cublas_status = cublasSetStream(cublas_handle_, default_stream_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            LOG_WARN("[NvidiaDeviceContext] cublasSetStream failed: " << cublas_status);
            // Non-fatal - continue with default stream
        }

        // Step 4: Query device name
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, device_ordinal_));
        device_name_ = props.name;

        LOG_DEBUG("[NvidiaDeviceContext] CUDA context initialized for device " << device_ordinal_
                                                                               << " (" << device_name_ << ")");
        LOG_TRACE("[NvidiaDeviceContext] - Default stream: " << default_stream_);
        LOG_TRACE("[NvidiaDeviceContext] - cuBLAS handle: " << cublas_handle_);

        // Step 5: Create cuBLASLt handle (for fused GEMM operations)
        cublasStatus_t cublas_lt_status = cublasLtCreate(&cublas_lt_handle_);
        if (cublas_lt_status != CUBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[NvidiaDeviceContext] cublasLtCreate failed: " << cublas_lt_status);
            return false;
        }
        LOG_TRACE("[NvidiaDeviceContext] - cuBLASLt handle: " << cublas_lt_handle_);

        /*
         * Inter-stream graph handoffs are part of the correctness contract, so
         * their event resources belong to the device context rather than to an
         * individual replay. Preallocating the complete pool here removes
         * cudaEventCreate/cudaEventDestroy from every decode transaction.
         */
        for (auto &event : stream_dependency_events_)
        {
            cudaError_t event_status =
                cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (event_status != cudaSuccess)
            {
                LOG_ERROR("[NvidiaDeviceContext] Failed to create persistent stream "
                          "dependency event: "
                          << cudaGetErrorString(event_status));
                cleanupOnWorker();
                return false;
            }
        }
        LOG_TRACE("[NvidiaDeviceContext] - Stream dependency events: "
                  << stream_dependency_events_.size());

        return true;
    }

    void NvidiaDeviceContext::cleanupOnWorker()
    {
        LOG_TRACE("[NvidiaDeviceContext] Cleaning up CUDA resources for device " << device_ordinal_);

        // Destroy cuBLASLt handle
        if (cublas_lt_handle_ != nullptr)
        {
            cublasStatus_t status = cublasLtDestroy(cublas_lt_handle_);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                LOG_WARN("[NvidiaDeviceContext] cublasLtDestroy failed: " << status);
            }
            cublas_lt_handle_ = nullptr;
        }

        // Destroy cuBLAS handle
        if (cublas_handle_ != nullptr)
        {
            cublasStatus_t status = cublasDestroy(cublas_handle_);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                LOG_WARN("[NvidiaDeviceContext] cublasDestroy failed: " << status);
            }
            cublas_handle_ = nullptr;
        }

        resetAuxiliaryStreams();

        /*
         * Every queued record/wait pair has finished by the time the worker
         * reaches context cleanup. Destroy the persistent handoff events once,
         * after graph owners have stopped and before their streams disappear.
         */
        for (auto &event : stream_dependency_events_)
        {
            if (event == nullptr)
                continue;

            cudaError_t err = cudaEventDestroy(event);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading)
            {
                LOG_ERROR("[NvidiaDeviceContext] cudaEventDestroy(stream dependency) "
                          "failed: "
                          << cudaGetErrorString(err));
            }
            event = nullptr;
        }

        // Destroy default stream (ignore cudaErrorCudartUnloading during shutdown)
        if (default_stream_ != nullptr)
        {
            cudaError_t err = cudaStreamDestroy(default_stream_);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading)
            {
                LOG_ERROR("[NvidiaDeviceContext] cudaStreamDestroy(default_stream_) failed: "
                          << cudaGetErrorString(err));
            }
            default_stream_ = nullptr;
        }

        // Note: We intentionally do NOT release the primary context here.
        // The CUDA driver manages the primary context lifetime, and releasing
        // it could cause issues if other code is still using it. The context
        // will be cleaned up when the process exits or when the last reference
        // is released.

        LOG_TRACE("[NvidiaDeviceContext] CUDA cleanup complete for device " << device_ordinal_);
    }

    // ============================================================================
    // Work Submission (Thread-Safe)
    // ============================================================================

    void NvidiaDeviceContext::submitAndWait(std::function<void()> work)
    {
        auto future = submitAsync(std::move(work));
        future.wait();
    }

    std::future<void> NvidiaDeviceContext::submitAsync(std::function<void()> work)
    {
        std::packaged_task<void()> task(std::move(work));
        std::future<void> future = task.get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);

            if (!running_.load())
            {
                // Worker not running - return a ready future with exception
                std::promise<void> p;
                p.set_exception(std::make_exception_ptr(
                    std::runtime_error("Worker thread not running")));
                return p.get_future();
            }

            work_queue_.push(std::move(task));
        }

        queue_cv_.notify_one();
        return future;
    }

    // ============================================================================
    // Stream Access (Worker-Thread-Only)
    // ============================================================================

    void *NvidiaDeviceContext::defaultStream()
    {
        return static_cast<void *>(default_stream_);
    }

    void *NvidiaDeviceContext::createStream()
    {
        // Create a NON-BLOCKING stream. This is required for stream capture (GPU graphs)
        // because CUTLASS/cuBLAS may internally dispatch to the legacy default stream.
        // A blocking stream would create illegal cross-stream dependencies during capture
        // (the blocking stream implicitly syncs with stream 0, violating capture mode).

        // Ensure we're on the correct device — createStream() may be called from
        // any thread, and the caller's active device may differ from device_ordinal_.
        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in createStream: " << cudaGetErrorString(set_err));
            return nullptr;
        }

        cudaStream_t stream;
        cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaStreamCreate failed: " << cudaGetErrorString(err));
            return nullptr;
        }
        return static_cast<void *>(stream);
    }

    void NvidiaDeviceContext::destroyStream(void *stream)
    {
        if (stream == nullptr)
        {
            return;
        }

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // Don't destroy the default stream
        if (cuda_stream == default_stream_)
        {
            LOG_WARN("[NvidiaDeviceContext] Attempted to destroy default stream - ignored");
            return;
        }

        CUDA_CHECK_VOID(cudaStreamDestroy(cuda_stream));
    }

    void *NvidiaDeviceContext::getOrCreateAuxiliaryStream(const std::string &name, bool *created)
    {
        if (created)
            *created = false;
        if (name.empty())
        {
            LOG_ERROR("[NvidiaDeviceContext] Auxiliary stream name must not be empty");
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(auxiliary_streams_mutex_);
        auto it = auxiliary_streams_.find(name);
        if (it != auxiliary_streams_.end() && it->second)
            return static_cast<void *>(it->second);

        auto *stream = static_cast<cudaStream_t>(createStream());
        if (!stream)
            return nullptr;
        auxiliary_streams_[name] = stream;
        if (created)
            *created = true;
        return static_cast<void *>(stream);
    }

    void NvidiaDeviceContext::resetAuxiliaryStreams()
    {
        std::lock_guard<std::mutex> lock(auxiliary_streams_mutex_);
        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            if (set_err == cudaErrorCudartUnloading)
            {
                auxiliary_streams_.clear();
                return;
            }
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in resetAuxiliaryStreams: "
                                                             << cudaGetErrorString(set_err));
            return;
        }
        for (auto &[name, stream] : auxiliary_streams_)
        {
            (void)name;
            if (stream && stream != default_stream_)
            {
                cudaError_t err = cudaStreamDestroy(stream);
                if (err != cudaSuccess && err != cudaErrorCudartUnloading)
                {
                    LOG_ERROR("[NvidiaDeviceContext] cudaStreamDestroy(auxiliary) failed: "
                              << cudaGetErrorString(err));
                }
            }
        }
        auxiliary_streams_.clear();
    }

    // ============================================================================
    // Event Access (Worker-Thread-Only)
    // ============================================================================

    void *NvidiaDeviceContext::createEvent()
    {
        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in createEvent: " << cudaGetErrorString(set_err));
            return nullptr;
        }

        cudaEvent_t event;
        cudaError_t err = cudaEventCreate(&event);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaEventCreate failed: " << cudaGetErrorString(err));
            return nullptr;
        }
        return static_cast<void *>(event);
    }

    void NvidiaDeviceContext::destroyEvent(void *event)
    {
        if (event == nullptr)
        {
            return;
        }

        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in destroyEvent: " << cudaGetErrorString(set_err));
            return;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        CUDA_CHECK_VOID(cudaEventDestroy(cuda_event));
    }

    void NvidiaDeviceContext::recordEvent(void *event, void *stream)
    {
        if (event == nullptr)
        {
            LOG_WARN("[NvidiaDeviceContext] recordEvent called with null event");
            return;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : default_stream_;

        CUDA_CHECK_VOID(cudaEventRecord(cuda_event, cuda_stream));
    }

    bool NvidiaDeviceContext::recordEventChecked(void *event, void *stream)
    {
        if (!event || !stream)
        {
            LOG_ERROR("[NvidiaDeviceContext] recordEventChecked requires non-null event and stream");
            return false;
        }

        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in recordEventChecked: " << cudaGetErrorString(set_err));
            return false;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaEventRecord(cuda_event, cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaEventRecord failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    void NvidiaDeviceContext::waitEvent(void *event, void *stream)
    {
        if (event == nullptr)
        {
            LOG_WARN("[NvidiaDeviceContext] waitEvent called with null event");
            return;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : default_stream_;

        // Make the stream wait for the event (GPU-side wait, not CPU blocking)
        CUDA_CHECK_VOID(cudaStreamWaitEvent(cuda_stream, cuda_event, 0));
    }

    bool NvidiaDeviceContext::waitEventChecked(void *event, void *stream)
    {
        if (!event || !stream)
        {
            LOG_ERROR("[NvidiaDeviceContext] waitEventChecked requires non-null event and stream");
            return false;
        }

        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in waitEventChecked: " << cudaGetErrorString(set_err));
            return false;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaStreamWaitEvent(cuda_stream, cuda_event, 0);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaStreamWaitEvent failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool NvidiaDeviceContext::queryEventChecked(void *event, bool &ready)
    {
        ready = false;
        if (!event)
        {
            LOG_ERROR("[NvidiaDeviceContext] queryEventChecked requires non-null event");
            return false;
        }

        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in queryEventChecked: " << cudaGetErrorString(set_err));
            return false;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        cudaError_t err = cudaEventQuery(cuda_event);
        if (err == cudaSuccess)
        {
            ready = true;
            return true;
        }
        if (err == cudaErrorNotReady)
        {
            return true;
        }

        LOG_ERROR("[NvidiaDeviceContext] cudaEventQuery failed: "
                  << cudaGetErrorString(err));
        return false;
    }

    void NvidiaDeviceContext::synchronizeEvent(void *event)
    {
        (void)synchronizeEventChecked(event);
    }

    bool NvidiaDeviceContext::synchronizeEventChecked(void *event)
    {
        if (event == nullptr)
        {
            LOG_ERROR("[NvidiaDeviceContext] synchronizeEventChecked called with null event");
            return false;
        }

        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in synchronizeEventChecked: "
                                                             << cudaGetErrorString(set_err));
            return false;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        cudaError_t err = cudaEventSynchronize(cuda_event);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaEventSynchronize failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    float NvidiaDeviceContext::eventElapsedTime(void *start, void *stop)
    {
        if (!start || !stop)
        {
            LOG_WARN("[NvidiaDeviceContext] eventElapsedTime called with null event");
            return -1.0f;
        }

        cudaEvent_t cuda_start = static_cast<cudaEvent_t>(start);
        cudaEvent_t cuda_stop = static_cast<cudaEvent_t>(stop);
        float ms = 0.0f;
        cudaError_t err = cudaEventElapsedTime(&ms, cuda_start, cuda_stop);
        if (err != cudaSuccess)
        {
            LOG_WARN("[NvidiaDeviceContext] cudaEventElapsedTime failed: " << cudaGetErrorString(err));
            return -1.0f;
        }
        return ms;
    }

    // ============================================================================
    // Library Handles (Worker-Thread-Only)
    // ============================================================================

    void *NvidiaDeviceContext::blasHandle()
    {
        return static_cast<void *>(cublas_handle_);
    }

    void *NvidiaDeviceContext::blasLtHandle()
    {
        return static_cast<void *>(cublas_lt_handle_);
    }

    // ============================================================================
    // Collective Communicator
    // ============================================================================

    void NvidiaDeviceContext::setCollectiveComm(void *comm)
    {
        nccl_comm_ = comm;
        LOG_TRACE("[NvidiaDeviceContext] Set NCCL communicator for device " << device_ordinal_);
    }

    void *NvidiaDeviceContext::collectiveComm() const
    {
        return nccl_comm_;
    }

    // ============================================================================
    // Synchronization (Thread-Safe)
    // ============================================================================

    void NvidiaDeviceContext::synchronize()
    {
        (void)synchronizeChecked();
    }

    bool NvidiaDeviceContext::synchronizeChecked()
    {
        bool ok = true;
        submitAndWait([this, &ok]()
                      {
        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess) {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                      << ") failed in synchronize: " << cudaGetErrorString(set_err));
            ok = false;
            return;
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("[NvidiaDeviceContext] cudaDeviceSynchronize failed: " 
                      << cudaGetErrorString(err));
            ok = false;
        } });
        return ok;
    }

    void NvidiaDeviceContext::synchronizeStream(void *stream)
    {
        (void)synchronizeStreamChecked(stream);
    }

    bool NvidiaDeviceContext::synchronizeStreamChecked(void *stream)
    {
        // Synchronize a specific stream. nullptr = legacy default stream (stream 0).
        // This is ~10× cheaper than cudaDeviceSynchronize() since it only waits
        // for one stream's work, not all streams on the device.
        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in synchronizeStreamChecked: " << cudaGetErrorString(set_err));
            return false;
        }
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : cudaStream_t(0);
        cudaError_t err = cudaStreamSynchronize(cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaStreamSynchronize failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool NvidiaDeviceContext::insertStreamDependency(
        void *dependent_stream,
        void *dependency_stream)
    {
        /*
         * A stream cannot depend on itself: all work within one stream is
         * already ordered. Avoid consuming a pool slot for this common no-op.
         */
        if (dependent_stream == dependency_stream)
            return true;

        /*
         * CUDA runtime device selection is thread-local. This method is called
         * directly from graph owner threads, not only from the context worker,
         * so select this context's device before touching its persistent event.
         */
        cudaError_t set_status = cudaSetDevice(device_ordinal_);
        if (set_status != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                      << ") failed in insertStreamDependency: "
                      << cudaGetErrorString(set_status));
            return false;
        }

        /*
         * nullptr retains the interface's legacy-stream meaning. Production
         * graph paths pass explicit streams, but preserving the mapping here
         * keeps the primitive well-defined for diagnostics and tests.
         */
        cudaStream_t dep_stream = dependent_stream ? static_cast<cudaStream_t>(dependent_stream) : cudaStream_t(0);
        cudaStream_t src_stream = dependency_stream ? static_cast<cudaStream_t>(dependency_stream) : cudaStream_t(0);

        /*
         * Round-robin selection prevents one busy graph owner from repeatedly
         * contending with another. The per-slot flag covers the short host-side
         * cudaEventRecord/cudaStreamWaitEvent pair only; the slot may be reused
         * as soon as the wait has been enqueued because CUDA stream waits bind
         * to the most recent record visible when cudaStreamWaitEvent is called.
         */
        const size_t event_index =
            next_stream_dependency_event_.fetch_add(
                1,
                std::memory_order_relaxed) %
            kStreamDependencyEventCount;
        std::atomic_flag &in_use =
            stream_dependency_event_in_use_[event_index];
        while (in_use.test_and_set(std::memory_order_acquire))
            std::this_thread::yield();

        struct ScopedEventLease final
        {
            std::atomic_flag &flag;
            ~ScopedEventLease()
            {
                flag.clear(std::memory_order_release);
            }
        } lease{in_use};

        cudaEvent_t event = stream_dependency_events_[event_index];
        if (event == nullptr)
        {
            LOG_ERROR("[NvidiaDeviceContext] Persistent stream dependency event "
                      << event_index << " is not initialized");
            return false;
        }

        cudaError_t err = cudaEventRecord(event, src_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaEventRecord failed in "
                      "insertStreamDependency: "
                      << cudaGetErrorString(err));
            return false;
        }

        err = cudaStreamWaitEvent(dep_stream, event, 0);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaStreamWaitEvent failed in "
                      "insertStreamDependency: "
                      << cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    std::unique_ptr<IGPUGraphCapture> NvidiaDeviceContext::createGraphCapture()
    {
        return std::make_unique<CUDAGraphCapture>(static_cast<cudaStream_t>(defaultStream()));
    }

    std::unique_ptr<IGPUGraphCapture> NvidiaDeviceContext::createGraphCapture(void *stream)
    {
        return std::make_unique<CUDAGraphCapture>(static_cast<cudaStream_t>(stream));
    }

    PointerValidationResult NvidiaDeviceContext::validatePointerDevice(const void *gpu_ptr, int expected_ordinal)
    {
        if (!gpu_ptr)
        {
            return {};
        }

        cudaPointerAttributes attr{};
        const cudaError_t err = cudaPointerGetAttributes(&attr, gpu_ptr);
        if (err != cudaSuccess)
        {
            cudaGetLastError();
            std::ostringstream oss;
            oss << "cudaPointerGetAttributes failed: " << cudaGetErrorString(err);
            return {false, -1, oss.str()};
        }

        const bool device_ok =
            (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) &&
            attr.device == expected_ordinal;
        if (device_ok)
        {
            return {true, attr.device, ""};
        }

        std::ostringstream oss;
        oss << "attr.device=" << attr.device
            << " expected=" << expected_ordinal
            << " attr.type=" << static_cast<int>(attr.type);
        if (attr.devicePointer)
        {
            oss << " device_ptr=" << attr.devicePointer;
        }
        if (attr.hostPointer)
        {
            oss << " host_ptr=" << attr.hostPointer;
        }

        return {false, attr.device, oss.str()};
    }

    PointerInspectionResult NvidiaDeviceContext::inspectPointer(const void *gpu_ptr) const
    {
        PointerInspectionResult result;
        if (!gpu_ptr)
        {
            return result;
        }

        cudaPointerAttributes attr{};
        const cudaError_t err = cudaPointerGetAttributes(&attr, gpu_ptr);
        if (err != cudaSuccess)
        {
            cudaGetLastError();
            return result;
        }

        result.known = true;
        result.actual_device = attr.device;
        result.base_ptr = attr.devicePointer ? attr.devicePointer : gpu_ptr;

        std::ostringstream oss;
        oss << "attr.device=" << attr.device
            << " attr.type=" << static_cast<int>(attr.type);
        if (attr.devicePointer)
        {
            oss << " device_ptr=" << attr.devicePointer;
        }
        if (attr.hostPointer)
        {
            oss << " host_ptr=" << attr.hostPointer;
        }
        result.details = oss.str();
        return result;
    }

    bool NvidiaDeviceContext::debugSynchronize()
    {
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaDeviceSynchronize failed during debug sync: "
                      << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // namespace llaminar2
