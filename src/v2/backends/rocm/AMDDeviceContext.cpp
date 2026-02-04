/**
 * @file AMDDeviceContext.cpp
 * @brief ROCm/HIP device context implementation with dedicated worker thread
 *
 * This file implements the AMDDeviceContext class, which provides:
 * - Dedicated worker thread for all HIP operations
 * - HIP context initialization and management
 * - hipBLAS handle creation and management
 * - Stream and event creation/destruction
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "AMDDeviceContext.h"
#include "../../utils/Logger.h"
#include <sstream>
#include <stdexcept>

namespace llaminar2
{

    // ============================================================================
    // Helper Macros for HIP Error Checking
    // ============================================================================

#define HIP_CHECK(call)                                                 \
    do                                                                  \
    {                                                                   \
        hipError_t err = (call);                                        \
        if (err != hipSuccess)                                          \
        {                                                               \
            LOG_ERROR("[AMDDeviceContext] " << #call << " failed: "     \
                                            << hipGetErrorString(err)); \
            return false;                                               \
        }                                                               \
    } while (0)

#define HIP_CHECK_VOID(call)                                            \
    do                                                                  \
    {                                                                   \
        hipError_t err = (call);                                        \
        if (err != hipSuccess)                                          \
        {                                                               \
            LOG_ERROR("[AMDDeviceContext] " << #call << " failed: "     \
                                            << hipGetErrorString(err)); \
        }                                                               \
    } while (0)

#define HIPBLAS_CHECK(call)                                         \
    do                                                              \
    {                                                               \
        hipblasStatus_t status = (call);                            \
        if (status != HIPBLAS_STATUS_SUCCESS)                       \
        {                                                           \
            LOG_ERROR("[AMDDeviceContext] " << #call << " failed: " \
                                            << status);             \
            return false;                                           \
        }                                                           \
    } while (0)

    // ============================================================================
    // Constructor / Destructor
    // ============================================================================

    AMDDeviceContext::AMDDeviceContext(int device_ordinal)
        : device_ordinal_(device_ordinal)
    {
        LOG_DEBUG("[AMDDeviceContext] Creating context for ROCm device " << device_ordinal);

        // Validate device ordinal before starting worker
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess)
        {
            std::ostringstream oss;
            oss << "hipGetDeviceCount failed: " << hipGetErrorString(err);
            throw std::runtime_error(oss.str());
        }

        if (device_ordinal < 0 || device_ordinal >= device_count)
        {
            std::ostringstream oss;
            oss << "Invalid ROCm device ordinal " << device_ordinal
                << " (available: 0-" << (device_count - 1) << ")";
            throw std::runtime_error(oss.str());
        }

        // Start worker thread
        shutdown_requested_.store(false);
        running_.store(false);

        try
        {
            worker_thread_ = std::thread(&AMDDeviceContext::workerLoop, this);

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
                throw std::runtime_error("Failed to initialize HIP context on worker thread");
            }

            LOG_INFO("[AMDDeviceContext] Created context for ROCm device " << device_ordinal
                                                                           << " (" << device_name_ << ")");
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[AMDDeviceContext] Failed to start worker thread: " << e.what());
            throw;
        }
    }

    AMDDeviceContext::~AMDDeviceContext()
    {
        LOG_DEBUG("[AMDDeviceContext] Destroying context for ROCm device " << device_ordinal_);

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

        LOG_DEBUG("[AMDDeviceContext] Context destroyed for ROCm device " << device_ordinal_);
    }

    // ============================================================================
    // Worker Thread Implementation
    // ============================================================================

    void AMDDeviceContext::workerLoop()
    {
        LOG_TRACE("[AMDDeviceContext] Worker thread starting for device " << device_ordinal_);

        // Initialize HIP context on this thread
        if (!initializeOnWorker())
        {
            LOG_ERROR("[AMDDeviceContext] Failed to initialize HIP context on worker thread");
            shutdown_requested_.store(true);
            queue_cv_.notify_all();
            return;
        }

        // Signal that initialization is complete
        initialized_.store(true);
        running_.store(true);
        queue_cv_.notify_all();

        LOG_TRACE("[AMDDeviceContext] Worker loop started for device " << device_ordinal_);

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
                    LOG_ERROR("[AMDDeviceContext] Exception in worker task: " << e.what());
                }
                catch (...)
                {
                    LOG_ERROR("[AMDDeviceContext] Unknown exception in worker task");
                }
            }
        }

        // Cleanup before exiting
        cleanupOnWorker();
        running_.store(false);

        LOG_TRACE("[AMDDeviceContext] Worker loop exited for device " << device_ordinal_);
    }

    bool AMDDeviceContext::initializeOnWorker()
    {
        LOG_TRACE("[AMDDeviceContext] Initializing HIP on worker thread for device "
                  << device_ordinal_);

        // Step 1: Set HIP device
        HIP_CHECK(hipSetDevice(device_ordinal_));

        // Step 2: Create default stream (non-blocking for better concurrency)
        HIP_CHECK(hipStreamCreateWithFlags(&default_stream_, hipStreamNonBlocking));

        // Step 3: Create hipBLAS handle
        hipblasStatus_t hipblas_status = hipblasCreate(&hipblas_handle_);
        if (hipblas_status != HIPBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[AMDDeviceContext] hipblasCreate failed: " << hipblas_status);
            return false;
        }

        // Associate hipBLAS handle with our stream
        hipblas_status = hipblasSetStream(hipblas_handle_, default_stream_);
        if (hipblas_status != HIPBLAS_STATUS_SUCCESS)
        {
            LOG_WARN("[AMDDeviceContext] hipblasSetStream failed: " << hipblas_status);
            // Non-fatal - continue with default stream
        }

        // Step 4: Query device name
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_ordinal_));
        device_name_ = props.name;

        LOG_DEBUG("[AMDDeviceContext] HIP context initialized for device " << device_ordinal_
                                                                           << " (" << device_name_ << ")");
        LOG_TRACE("[AMDDeviceContext] - Default stream: " << default_stream_);
        LOG_TRACE("[AMDDeviceContext] - hipBLAS handle: " << hipblas_handle_);

        // Step 5: Create hipBLASLt handle (for fused GEMM operations)
        hipblasStatus_t hipblas_lt_status = hipblasLtCreate(&hipblas_lt_handle_);
        if (hipblas_lt_status != HIPBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[AMDDeviceContext] hipblasLtCreate failed: " << hipblas_lt_status);
            return false;
        }
        LOG_TRACE("[AMDDeviceContext] - hipBLASLt handle: " << hipblas_lt_handle_);

        return true;
    }

    void AMDDeviceContext::cleanupOnWorker()
    {
        LOG_TRACE("[AMDDeviceContext] Cleaning up HIP resources for device " << device_ordinal_);

        // Destroy hipBLASLt handle
        if (hipblas_lt_handle_ != nullptr)
        {
            hipblasStatus_t status = hipblasLtDestroy(hipblas_lt_handle_);
            if (status != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_WARN("[AMDDeviceContext] hipblasLtDestroy failed: " << status);
            }
            hipblas_lt_handle_ = nullptr;
        }

        // Destroy hipBLAS handle
        if (hipblas_handle_ != nullptr)
        {
            hipblasStatus_t status = hipblasDestroy(hipblas_handle_);
            if (status != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_WARN("[AMDDeviceContext] hipblasDestroy failed: " << status);
            }
            hipblas_handle_ = nullptr;
        }

        // Destroy default stream
        if (default_stream_ != nullptr)
        {
            HIP_CHECK_VOID(hipStreamDestroy(default_stream_));
            default_stream_ = nullptr;
        }

        LOG_TRACE("[AMDDeviceContext] HIP cleanup complete for device " << device_ordinal_);
    }

    // ============================================================================
    // Work Submission (Thread-Safe)
    // ============================================================================

    void AMDDeviceContext::submitAndWait(std::function<void()> work)
    {
        auto future = submitAsync(std::move(work));
        future.wait();
    }

    std::future<void> AMDDeviceContext::submitAsync(std::function<void()> work)
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

    void *AMDDeviceContext::defaultStream()
    {
        return static_cast<void *>(default_stream_);
    }

    void *AMDDeviceContext::createStream()
    {
        hipStream_t stream;
        hipError_t err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipStreamCreate failed: " << hipGetErrorString(err));
            return nullptr;
        }
        return static_cast<void *>(stream);
    }

    void AMDDeviceContext::destroyStream(void *stream)
    {
        if (stream == nullptr)
        {
            return;
        }

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        // Don't destroy the default stream
        if (hip_stream == default_stream_)
        {
            LOG_WARN("[AMDDeviceContext] Attempted to destroy default stream - ignored");
            return;
        }

        HIP_CHECK_VOID(hipStreamDestroy(hip_stream));
    }

    // ============================================================================
    // Event Access (Worker-Thread-Only)
    // ============================================================================

    void *AMDDeviceContext::createEvent()
    {
        hipEvent_t event;
        hipError_t err = hipEventCreate(&event);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipEventCreate failed: " << hipGetErrorString(err));
            return nullptr;
        }
        return static_cast<void *>(event);
    }

    void AMDDeviceContext::destroyEvent(void *event)
    {
        if (event == nullptr)
        {
            return;
        }

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        HIP_CHECK_VOID(hipEventDestroy(hip_event));
    }

    void AMDDeviceContext::recordEvent(void *event, void *stream)
    {
        if (event == nullptr)
        {
            LOG_WARN("[AMDDeviceContext] recordEvent called with null event");
            return;
        }

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        hipStream_t hip_stream = stream ? static_cast<hipStream_t>(stream) : default_stream_;

        HIP_CHECK_VOID(hipEventRecord(hip_event, hip_stream));
    }

    void AMDDeviceContext::waitEvent(void *event, void *stream)
    {
        if (event == nullptr)
        {
            LOG_WARN("[AMDDeviceContext] waitEvent called with null event");
            return;
        }

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        hipStream_t hip_stream = stream ? static_cast<hipStream_t>(stream) : default_stream_;

        // Make the stream wait for the event (GPU-side wait, not CPU blocking)
        HIP_CHECK_VOID(hipStreamWaitEvent(hip_stream, hip_event, 0));
    }

    void AMDDeviceContext::synchronizeEvent(void *event)
    {
        if (event == nullptr)
        {
            LOG_WARN("[AMDDeviceContext] synchronizeEvent called with null event");
            return;
        }

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        HIP_CHECK_VOID(hipEventSynchronize(hip_event));
    }

    // ============================================================================
    // Library Handles (Worker-Thread-Only)
    // ============================================================================

    void *AMDDeviceContext::blasHandle()
    {
        return static_cast<void *>(hipblas_handle_);
    }

    void *AMDDeviceContext::blasLtHandle()
    {
        return static_cast<void *>(hipblas_lt_handle_);
    }

    // ============================================================================
    // Collective Communicator
    // ============================================================================

    void AMDDeviceContext::setCollectiveComm(void *comm)
    {
        rccl_comm_ = comm;
        LOG_TRACE("[AMDDeviceContext] Set RCCL communicator for device " << device_ordinal_);
    }

    void *AMDDeviceContext::collectiveComm() const
    {
        return rccl_comm_;
    }

    // ============================================================================
    // Synchronization (Thread-Safe)
    // ============================================================================

    void AMDDeviceContext::synchronize()
    {
        submitAndWait([this]()
                      {
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            LOG_ERROR("[AMDDeviceContext] hipDeviceSynchronize failed: " 
                      << hipGetErrorString(err));
        } });
    }

} // namespace llaminar2
