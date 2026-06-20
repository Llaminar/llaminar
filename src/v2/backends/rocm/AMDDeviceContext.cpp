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
#include "HIPGraphCapture.h"
#include "ROCmBackend.h"
#include "../../utils/Logger.h"
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        bool setAMDDeviceForResource(int device_ordinal, const char *context)
        {
            hipError_t err = hipSetDevice(device_ordinal);
            if (err != hipSuccess)
            {
                LOG_ERROR("[AMDDeviceContext] hipSetDevice(" << device_ordinal
                                                             << ") failed in " << context
                                                             << ": " << hipGetErrorString(err));
                return false;
            }
            return true;
        }
    } // namespace

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

            LOG_DEBUG("[AMDDeviceContext] Created context for ROCm device " << device_ordinal
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

        // Step 2: Create default stream.
        // IMPORTANT: Use hipStreamDefault (blocking) instead of hipStreamNonBlocking.
        // Non-blocking streams do NOT synchronize with the legacy default stream (stream 0/nullptr).
        // Some ROCm kernels (hipBLAS, CK) may internally dispatch work to stream 0,
        // causing data races when the main compute stream is non-blocking.
        // A blocking stream maintains implicit synchronization with stream 0,
        // preventing these races while still allowing graph capture.
        HIP_CHECK(hipStreamCreateWithFlags(&default_stream_, hipStreamDefault));

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

        // Disable atomic reductions inside rocBLAS/Tensile kernels.  Atomic
        // reductions produce nondeterministic FP accumulation order across
        // runs, which causes parity tests to flake (~14x KL divergence
        // spread on gfx906).  The non-atomic Tensile paths are selected
        // instead; perf impact is minor since SGEMM on gfx906 is
        // compute-bound rather than reduction-bound.
        hipblas_status = hipblasSetAtomicsMode(hipblas_handle_, HIPBLAS_ATOMICS_NOT_ALLOWED);
        if (hipblas_status != HIPBLAS_STATUS_SUCCESS)
        {
            LOG_WARN("[AMDDeviceContext] hipblasSetAtomicsMode failed: " << hipblas_status);
            // Non-fatal - results may be nondeterministic but correct
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

        // Guard: Check if HIP runtime is still alive before cleanup.
        // During static destruction (process exit), the HIP runtime's internal
        // state may already be torn down by atexit handlers. Calling HIP APIs
        // on dead runtime state causes SIGSEGV (use-after-free at cmpb 0xb8(%rdi)
        // inside hipHccModuleLaunchKernel). This matches CUDA's approach of
        // ignoring cudaErrorCudartUnloading during NvidiaDeviceContext cleanup.
        int device_count = 0;
        hipError_t probe = hipGetDeviceCount(&device_count);
        if (probe != hipSuccess)
        {
            LOG_TRACE("[AMDDeviceContext] HIP runtime already shut down for device "
                      << device_ordinal_ << " (hipGetDeviceCount returned "
                      << hipGetErrorString(probe) << "), skipping cleanup");
            hipblas_lt_handle_ = nullptr;
            hipblas_handle_ = nullptr;
            default_stream_ = nullptr;
            return;
        }

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
        if (!setAMDDeviceForResource(device_ordinal_, "createStream"))
        {
            return nullptr;
        }

        hipStream_t stream;
        // Use non-blocking stream for graph capture compatibility.
        // HIP graph replay requires non-blocking streams to avoid implicit
        // serialization with the null stream that can cause incorrect results.
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

        if (!setAMDDeviceForResource(device_ordinal_, "destroyStream"))
        {
            return;
        }

        HIP_CHECK_VOID(hipStreamDestroy(hip_stream));
    }

    // ============================================================================
    // Event Access (Worker-Thread-Only)
    // ============================================================================

    void *AMDDeviceContext::createEvent()
    {
        if (!setAMDDeviceForResource(device_ordinal_, "createEvent"))
        {
            return nullptr;
        }

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
        if (!setAMDDeviceForResource(device_ordinal_, "destroyEvent"))
        {
            return;
        }
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

    float AMDDeviceContext::eventElapsedTime(void *start, void *stop)
    {
        if (!start || !stop)
        {
            LOG_WARN("[AMDDeviceContext] eventElapsedTime called with null event");
            return -1.0f;
        }

        hipEvent_t hip_start = static_cast<hipEvent_t>(start);
        hipEvent_t hip_stop = static_cast<hipEvent_t>(stop);
        float ms = 0.0f;
        hipError_t err = hipEventElapsedTime(&ms, hip_start, hip_stop);
        if (err != hipSuccess)
        {
            LOG_WARN("[AMDDeviceContext] hipEventElapsedTime failed: " << hipGetErrorString(err));
            return -1.0f;
        }
        return ms;
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
        // During graph capture, hipDeviceSynchronize() is illegal — it poisons
        // the capture state. Skip the sync entirely since no real GPU work is
        // happening during capture recording (kernels are just being recorded).
        if (capture_active_.load(std::memory_order_acquire))
        {
            return;
        }

        submitAndWait([this]()
                      {
        // Re-check capture_active_ inside worker — another thread's capture
        // controller may have started capture between our outer check and
        // this worker dispatch (race in multi-device TP graph capture).
        if (capture_active_.load(std::memory_order_acquire))
            return;

        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            // hipErrorStreamCaptureUnsupported (900): another thread started
            // graph capture on this device after our check. Benign race in
            // multi-device TP — the capture will complete without this sync.
            if (err == hipErrorStreamCaptureUnsupported ||
                err == hipErrorStreamCaptureImplicit) {
                LOG_DEBUG("[AMDDeviceContext] Skipping hipDeviceSynchronize during "
                          "concurrent graph capture (ordinal=" << device_ordinal_ << ")");
                return;
            }
            LOG_ERROR("[AMDDeviceContext] hipDeviceSynchronize failed: " 
                      << hipGetErrorString(err));
        } });
    }

    void AMDDeviceContext::synchronizeStream(void *stream)
    {
        (void)synchronizeStreamChecked(stream);
    }

    bool AMDDeviceContext::synchronizeStreamChecked(void *stream)
    {
        // During graph capture, stream sync is illegal on the capture stream.
        if (capture_active_.load(std::memory_order_acquire))
        {
            return true;
        }

        hipStream_t hip_stream = stream ? static_cast<hipStream_t>(stream) : hipStream_t(0);
        hipError_t err = hipStreamSynchronize(hip_stream);
        if (err != hipSuccess)
        {
            // Benign race: another thread started capture after our check.
            if (err == hipErrorStreamCaptureUnsupported ||
                err == hipErrorStreamCaptureImplicit)
            {
                LOG_DEBUG("[AMDDeviceContext] Skipping hipStreamSynchronize during "
                          "concurrent graph capture (ordinal="
                          << device_ordinal_ << ")");
                return true;
            }
            LOG_ERROR("[AMDDeviceContext] hipStreamSynchronize failed: "
                      << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    void AMDDeviceContext::insertStreamDependency(void *dependent_stream, void *dependency_stream)
    {
        // GPU-side inter-stream synchronization via events.
        // Records an event on dependency_stream, makes dependent_stream wait for it.
        // Unlike synchronizeStream(), this does NOT block the CPU.
        hipStream_t dep_stream = dependent_stream ? static_cast<hipStream_t>(dependent_stream) : hipStream_t(0);
        hipStream_t src_stream = dependency_stream ? static_cast<hipStream_t>(dependency_stream) : hipStream_t(0);

        // hipEventCreate associates the event with the "current" device on the
        // calling thread.  When called from multi-device TP replay threads the
        // thread-local device may not match this context's device, so force it.
        hipError_t set_err = hipSetDevice(device_ordinal_);
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipSetDevice(" << device_ordinal_
                                                         << ") failed in insertStreamDependency: " << hipGetErrorString(set_err));
            return;
        }

        hipEvent_t event;
        hipError_t err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipEventCreate failed in insertStreamDependency: "
                      << hipGetErrorString(err));
            return;
        }

        err = hipEventRecord(event, src_stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipEventRecord failed: " << hipGetErrorString(err));
            hipEventDestroy(event);
            return;
        }

        err = hipStreamWaitEvent(dep_stream, event, 0);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipStreamWaitEvent failed: " << hipGetErrorString(err));
        }

        hipEventDestroy(event);
    }

    std::unique_ptr<IGPUGraphCapture> AMDDeviceContext::createGraphCapture()
    {
        return std::make_unique<HIPGraphCapture>(static_cast<hipStream_t>(defaultStream()));
    }

    std::unique_ptr<IGPUGraphCapture> AMDDeviceContext::createGraphCapture(void *stream)
    {
        return std::make_unique<HIPGraphCapture>(static_cast<hipStream_t>(stream));
    }

    void AMDDeviceContext::clearLastError()
    {
        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            LOG_DEBUG("[AMDDeviceContext] Cleared sticky HIP error on device "
                      << device_ordinal_ << ": " << hipGetErrorString(err));
        }
    }

    PointerValidationResult AMDDeviceContext::validatePointerDevice(const void *gpu_ptr, int expected_ordinal)
    {
        if (!gpu_ptr)
        {
            return {};
        }

        hipPointerAttribute_t attr{};
        const hipError_t err = hipPointerGetAttributes(&attr, gpu_ptr);
        if (err != hipSuccess)
        {
            ROCmPointerOwnerInfo owner;
            const bool has_owner = ROCmBackend::queryPointerOwner(gpu_ptr, owner);

            std::ostringstream oss;
            oss << "hipPointerGetAttributes failed: " << hipGetErrorString(err);
            if (has_owner)
            {
                oss << " owner.dev=" << owner.device_id
                    << " owner.bytes=" << owner.size_bytes
                    << " owner.seq=" << owner.sequence;
            }
            else
            {
                oss << " owner=unknown";
            }
            return {false, -1, oss.str()};
        }

        if (attr.device == expected_ordinal)
        {
            return {true, attr.device, ""};
        }

        ROCmPointerOwnerInfo owner;
        const bool has_owner = ROCmBackend::queryPointerOwner(gpu_ptr, owner);

        std::ostringstream oss;
        oss << "attr.device=" << attr.device
            << " expected=" << expected_ordinal;
        if (has_owner)
        {
            oss << " owner.base=" << owner.base_ptr
                << " owner.dev=" << owner.device_id
                << " owner.bytes=" << owner.size_bytes
                << " owner.seq=" << owner.sequence
                << " owner.thread=" << owner.thread_hash
                << " owner.active=" << owner.active;
        }
        else
        {
            oss << " owner=unknown";
        }

        return {false, attr.device, oss.str()};
    }

    PointerInspectionResult AMDDeviceContext::inspectPointer(const void *gpu_ptr) const
    {
        PointerInspectionResult result;
        if (!gpu_ptr)
        {
            return result;
        }

        ROCmPointerOwnerInfo owner;
        if (ROCmBackend::queryPointerOwner(gpu_ptr, owner))
        {
            result.known = true;
            result.active = owner.active;
            result.actual_device = owner.device_id;
            result.base_ptr = owner.base_ptr;
            result.size_bytes = owner.size_bytes;
            result.sequence = owner.sequence;
            result.thread_hash = owner.thread_hash;

            std::ostringstream oss;
            oss << "owner_base=" << owner.base_ptr
                << " owner_bytes=" << owner.size_bytes
                << " owner_device=" << owner.device_id
                << " owner_seq=" << owner.sequence
                << " owner_thread=" << owner.thread_hash;
            result.details = oss.str();
            return result;
        }

        hipPointerAttribute_t attr{};
        const hipError_t err = hipPointerGetAttributes(&attr, gpu_ptr);
        if (err == hipSuccess)
        {
            result.known = true;
            result.actual_device = attr.device;

            std::ostringstream oss;
            oss << "attr.device=" << attr.device;
            result.details = oss.str();
        }

        return result;
    }

    void AMDDeviceContext::dumpRecentPointerEvents(size_t max_events)
    {
        ROCmBackend::dumpRecentPointerEvents(max_events);
    }

    bool AMDDeviceContext::debugSynchronize()
    {
        if (capture_active_.load(std::memory_order_acquire))
        {
            return true;
        }

        const hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipDeviceSynchronize failed during debug sync: "
                      << hipGetErrorString(sync_err));
            ROCmBackend::dumpRecentPointerEvents(96);
            return false;
        }

        return true;
    }

} // namespace llaminar2
