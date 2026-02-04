/**
 * @file NvidiaDeviceContext.cu
 * @brief CUDA device context implementation with dedicated worker thread
 *
 * This file implements the NvidiaDeviceContext class, which provides:
 * - Dedicated worker thread for all CUDA operations
 * - CUDA primary context retention via driver API
 * - cuBLAS handle creation and management
 * - Stream and event creation/destruction
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "NvidiaDeviceContext.h"
#include "../../utils/Logger.h"
#include <cuda.h> // For cuDevicePrimaryCtxRetain, cuCtxSetCurrent
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

        // Step 2: Retain primary context via driver API for robustness
        // This ensures the context persists even if the runtime API releases it
        CUdevice cu_device;
        CUresult cu_err = cuDeviceGet(&cu_device, device_ordinal_);
        if (cu_err != CUDA_SUCCESS)
        {
            LOG_WARN("[NvidiaDeviceContext] cuDeviceGet failed: " << cu_err
                                                                  << " (continuing with runtime API only)");
        }
        else
        {
            CUcontext ctx;
            cu_err = cuDevicePrimaryCtxRetain(&ctx, cu_device);
            if (cu_err != CUDA_SUCCESS)
            {
                LOG_WARN("[NvidiaDeviceContext] cuDevicePrimaryCtxRetain failed: " << cu_err);
            }
            else
            {
                cu_err = cuCtxSetCurrent(ctx);
                if (cu_err != CUDA_SUCCESS)
                {
                    LOG_WARN("[NvidiaDeviceContext] cuCtxSetCurrent failed: " << cu_err);
                }
                else
                {
                    cuda_context_ = ctx;
                    LOG_TRACE("[NvidiaDeviceContext] Retained CUDA primary context for device "
                              << device_ordinal_);
                }
            }
        }

        // Step 3: Create default stream (non-blocking for better concurrency)
        CUDA_CHECK(cudaStreamCreateWithFlags(&default_stream_, cudaStreamNonBlocking));

        // Step 4: Create cuBLAS handle
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

        // Step 5: Query device name
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, device_ordinal_));
        device_name_ = props.name;

        LOG_DEBUG("[NvidiaDeviceContext] CUDA context initialized for device " << device_ordinal_
                                                                               << " (" << device_name_ << ")");
        LOG_TRACE("[NvidiaDeviceContext] - Default stream: " << default_stream_);
        LOG_TRACE("[NvidiaDeviceContext] - cuBLAS handle: " << cublas_handle_);

        // Step 6: Create cuBLASLt handle (for fused GEMM operations)
        cublasStatus_t cublas_lt_status = cublasLtCreate(&cublas_lt_handle_);
        if (cublas_lt_status != CUBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[NvidiaDeviceContext] cublasLtCreate failed: " << cublas_lt_status);
            return false;
        }
        LOG_TRACE("[NvidiaDeviceContext] - cuBLASLt handle: " << cublas_lt_handle_);

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

        // Destroy default stream
        if (default_stream_ != nullptr)
        {
            CUDA_CHECK_VOID(cudaStreamDestroy(default_stream_));
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

    // ============================================================================
    // Event Access (Worker-Thread-Only)
    // ============================================================================

    void *NvidiaDeviceContext::createEvent()
    {
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

    void NvidiaDeviceContext::synchronizeEvent(void *event)
    {
        if (event == nullptr)
        {
            LOG_WARN("[NvidiaDeviceContext] synchronizeEvent called with null event");
            return;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        CUDA_CHECK_VOID(cudaEventSynchronize(cuda_event));
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
        submitAndWait([this]()
                      {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("[NvidiaDeviceContext] cudaDeviceSynchronize failed: " 
                      << cudaGetErrorString(err));
        } });
    }

} // namespace llaminar2
