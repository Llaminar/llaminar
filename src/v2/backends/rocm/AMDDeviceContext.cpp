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
 * Context setup retires hipBLAS's initial vendor-owned workspace before the
 * context is published. Every arithmetic submission subsequently binds its
 * admitted stage arena, so first-use capture never frees that initial pool.
 *
 * @author David Sanftenberg
 * @date February 2026
 * Hardware identity comes from the immutable process inventory, while native
 * streams, handles and runtime generations remain context-owned resources.
 * Concurrent projection side streams/events belong to this exact context and
 * cannot be reset while a complete native capture submission borrows them.
 */

#include "AMDDeviceContext.h"
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include "HIPGraphCapture.h"
#include "ROCmBackend.h"
#include "ROCmRuntimeStartup.h"
#include "ROCmHardwareMetadata.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/GPUWorkerInitialization.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
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
        : process_lifetime_lease_(GPUDeviceProcessLifetime::acquireNativeContext()),
          device_ordinal_(device_ordinal)
    {
        requireROCmRuntimeStartup();
        LOG_DEBUG("[AMDDeviceContext] Creating context for ROCm device " << device_ordinal);

        // Validate device ordinal before starting worker
        initializeROCmHardwareMetadata();
        const int device_count = static_cast<int>(rocmHardwareMetadata().size());

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
            {
                // The dedicated GPU-control child needs the admitted rank
                // partition, not its creator's first OpenMP place. Restore
                // the creator before waiting or executing any caller work.
                const ScopedOpenMPPartitionAffinity control_affinity;
                worker_thread_ = std::thread(&AMDDeviceContext::workerLoop, this);
            }

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
        /*
         * Publish exact ownership before any initialization helper can make a
         * nested synchronous submission. The constructor observes this write
         * through the initialized_ release/acquire handoff before exposing the
         * context to other threads.
         */
        worker_thread_id_ = std::this_thread::get_id();
        LOG_TRACE("[AMDDeviceContext] Worker thread starting for device " << device_ordinal_);

        // One owner-thread guard covers every unpublished handle and every
        // false/throw exit. Initialization never performs a second local cleanup.
        bool ready = false;
        try
        {
            ready = runGPUWorkerInitialization(
                [this] { return initializeOnWorker(); },
                [this] { cleanupOnWorker(); });
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[AMDDeviceContext] Worker initialization exception after rollback: " << error.what());
        }
        catch (...)
        {
            LOG_ERROR("[AMDDeviceContext] Unknown worker initialization exception after rollback");
        }
        if (!ready)
        {
            LOG_ERROR("[AMDDeviceContext] Failed to initialize HIP context on worker thread");
            publishGPUWorkerReadiness(queue_mutex_, queue_cv_, initialized_, running_,
                shutdown_requested_, GPUWorkerReadiness::Failed);
            return;
        }

        // Publish every ready flag under the constructor's predicate mutex.
        publishGPUWorkerReadiness(queue_mutex_, queue_cv_, initialized_, running_,
            shutdown_requested_, GPUWorkerReadiness::Ready);

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

    /**
     * @brief Create unpublished native resources on their owning device worker.
     * @return True after library handles, streams and dependency events are ready.
     *
     * Retire the initial rocBLAS allocation here, outside graph capture. A null
     * workspace leaves rocBLAS in its managed state with no allocation; it does
     * not disable vendor allocation globally. HipBLASGemmKernel must bind the
     * admitted non-null arena before every arithmetic submission.
     */
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

        // Replacing the initial vendor pool during first-use graph recording
        // would call hipFree inside capture. Retire it before publication; the
        // arithmetic adapters still require their real stage-owned workspace.
        hipblas_status = hipblasSetWorkspace(hipblas_handle_, nullptr, 0);
        if (hipblas_status != HIPBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[AMDDeviceContext] hipblasSetWorkspace(nullptr, 0) failed "
                      "while retiring the initial vendor allocation: " << hipblas_status);
            return false;
        }

        // Associate hipBLAS handle with our stream
        hipblas_status = hipblasSetStream(hipblas_handle_, default_stream_);
        if (hipblas_status != HIPBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[AMDDeviceContext] hipblasSetStream failed: " << hipblas_status);
            return false;
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
            LOG_ERROR("[AMDDeviceContext] hipblasSetAtomicsMode failed: " << hipblas_status);
            return false;
        }

        // Step 4: Query device name
        device_name_ = rocmHardwareMetadata().at(device_ordinal_).name;

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

        /*
         * Stream handoffs are issued on every graph replay. Own their events
         * for the entire device-context lifetime so replay never allocates or
         * destroys HIP runtime resources in the inference hot path.
         */
        for (auto &event : stream_dependency_events_)
        {
            hipError_t event_status =
                hipEventCreateWithFlags(&event, hipEventDisableTiming);
            if (event_status != hipSuccess)
            {
                LOG_ERROR("[AMDDeviceContext] Failed to create persistent stream "
                          "dependency event: "
                          << hipGetErrorString(event_status));
                return false;
            }
        }
        LOG_TRACE("[AMDDeviceContext] - Stream dependency events: "
                  << stream_dependency_events_.size());

        return true;
    }

    void AMDDeviceContext::cleanupOnWorker()
    {
        LOG_TRACE("[AMDDeviceContext] Cleaning up HIP resources for device " << device_ordinal_);
        // Reject live capture before destroying any context-generation resource.
        // Exact side handles retire while the selected runtime still owns them.
        projection_resources_.retire(*this);

        // Destroy hipBLASLt handle
        if (hipblas_lt_handle_ != nullptr)
        {
            hipblasStatus_t status = hipblasLtDestroy(hipblas_lt_handle_);
            if (status != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[AMDDeviceContext] hipblasLtDestroy failed: " << status);
                std::terminate();
            }
            hipblas_lt_handle_ = nullptr;
        }

        // Destroy hipBLAS handle
        if (hipblas_handle_ != nullptr)
        {
            hipblasStatus_t status = hipblasDestroy(hipblas_handle_);
            if (status != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[AMDDeviceContext] hipblasDestroy failed: " << status);
                std::terminate();
            }
            hipblas_handle_ = nullptr;
        }

        resetAuxiliaryStreams();

        /*
         * Context shutdown occurs after the worker queue has drained, so no
         * graph owner can still enqueue a record/wait pair. Destroy each
         * persistent handoff event exactly once before destroying streams.
         */
        for (auto &event : stream_dependency_events_)
        {
            if (event == nullptr)
                continue;

            hipError_t err = hipEventDestroy(event);
            if (err != hipSuccess)
            {
                LOG_ERROR("[AMDDeviceContext] hipEventDestroy(stream dependency) "
                          "failed: "
                          << hipGetErrorString(err));
                std::terminate();
            }
            event = nullptr;
        }

        // Destroy default stream
        if (default_stream_ != nullptr)
        {
            const auto error = hipStreamDestroy(default_stream_);
            if (error != hipSuccess)
            {
                LOG_ERROR("[AMDDeviceContext] hipStreamDestroy(default_stream_) device=" << device_ordinal_
                          << " handle=" << static_cast<void *>(default_stream_)
                          << " status=" << static_cast<int>(error) << " error=" << hipGetErrorString(error));
                std::terminate();
            }
            default_stream_ = nullptr;
        }

        LOG_TRACE("[AMDDeviceContext] HIP cleanup complete for device " << device_ordinal_);
    }

    // ============================================================================
    // Work Submission (Thread-Safe)
    // ============================================================================

    bool AMDDeviceContext::ownsCurrentThread() const noexcept
    {
        return worker_thread_id_ == std::this_thread::get_id();
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

    void *AMDDeviceContext::getOrCreateAuxiliaryStream(const std::string &name, bool *created)
    {
        return getOrCreateAuxiliaryStream(
            name,
            GPUAuxiliaryStreamSchedulingClass::Normal,
            created);
    }

    /**
     * @brief Materialize one HIP auxiliary stream with immutable priority.
     *
     * The highest-priority class is used by bounded controller work and the
     * lowest-priority class by residency maintenance. HIP copy-engine
     * scheduling is not inferred from this compute-stream priority.
     */
    void *AMDDeviceContext::getOrCreateAuxiliaryStream(
        const std::string &name,
        GPUAuxiliaryStreamSchedulingClass scheduling_class,
        bool *created)
    {
        if (created)
            *created = false;
        if (name.empty())
        {
            LOG_ERROR("[AMDDeviceContext] Auxiliary stream name must not be empty");
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(auxiliary_streams_mutex_);
        auto it = auxiliary_streams_.find(name);
        if (it != auxiliary_streams_.end() && it->second)
        {
            const auto class_it =
                auxiliary_stream_scheduling_classes_.find(name);
            if (class_it == auxiliary_stream_scheduling_classes_.end() ||
                class_it->second != scheduling_class)
            {
                LOG_ERROR("[AMDDeviceContext] Auxiliary stream scheduling class changed for "
                          << name);
                return nullptr;
            }
            return static_cast<void *>(it->second);
        }

        hipStream_t stream = nullptr;
        if (scheduling_class ==
            GPUAuxiliaryStreamSchedulingClass::Normal)
        {
            stream = static_cast<hipStream_t>(createStream());
        }
        else
        {
            if (!setAMDDeviceForResource(
                    device_ordinal_,
                    "getOrCreateAuxiliaryStream(prioritized)"))
            {
                return nullptr;
            }
            int least_priority = 0;
            int greatest_priority = 0;
            hipError_t error = hipDeviceGetStreamPriorityRange(
                &least_priority, &greatest_priority);
            const int priority =
                scheduling_class ==
                        GPUAuxiliaryStreamSchedulingClass::LatencyCritical
                    ? greatest_priority
                    : least_priority;
            if (error == hipSuccess)
            {
                error = hipStreamCreateWithPriority(
                    &stream,
                    hipStreamNonBlocking,
                    priority);
            }
            if (error != hipSuccess)
            {
                LOG_ERROR("[AMDDeviceContext] Could not create prioritized auxiliary stream: "
                          << hipGetErrorString(error));
                return nullptr;
            }
        }
        if (!stream)
            return nullptr;
        auxiliary_streams_[name] = stream;
        auxiliary_stream_scheduling_classes_[name] = scheduling_class;
        if (created)
            *created = true;
        return static_cast<void *>(stream);
    }

    /** @copydoc IWorkerGPUContext::prepareProjectionResources */
    void AMDDeviceContext::prepareProjectionResources()
    {
        if (!isInitialized()) throw std::logic_error("HIP projection preparation requires a live context");
        if (isGraphCaptureActive()) throw std::logic_error("HIP projection resources must be prepared before capture");
        if (!setAMDDeviceForResource(device_ordinal_, "prepareProjectionResources"))
            throw std::runtime_error("Cannot select exact HIP projection resource owner");
        // Replace the existing fixed pool at its original cold boundary. Native
        // queues/events remain under the existing opaque PMA admission scope.
        projection_resources_.prepare(*this);
    }
    /** @copydoc IWorkerGPUContext::acquireProjectionSubmission */
    GPUProjectionSubmission AMDDeviceContext::acquireProjectionSubmission(
        void *root, GPUProjectionSubmissionKind kind, bool native_capturing)
    {
        if (!isInitialized()) throw std::logic_error("HIP projection submission requires a live context");
        const auto stream = static_cast<hipStream_t>(root);
        hipDevice_t stream_device = -1;
        if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread ||
            hipSetDevice(device_ordinal_) != hipSuccess ||
            hipStreamGetDevice(stream, &stream_device) != hipSuccess || stream_device != device_ordinal_)
            throw std::logic_error("HIP projection requires an exact stream on its owning native device");
        if (kind == GPUProjectionSubmissionKind::Eager && !native_capturing)
            projection_resources_.prepare(*this); // Original eager cold boundary; never during native recording.
        auto submission = projection_resources_.acquire(root, kind, native_capturing);
        retainProjectionSubmission(submission);
        return submission;
    }
    /** @copydoc IWorkerGPUContext::assertProjectionResourcesRetirable */
    void AMDDeviceContext::assertProjectionResourcesRetirable()
    { projection_resources_.assertRetirable(); }
    /** @copydoc IWorkerGPUContext::beginProjectionResourcesRetirement */
    void AMDDeviceContext::beginProjectionResourcesRetirement()
    { projection_resources_.beginRetirement(); }
    /** @copydoc GPUProjectionResourceOperations::createProjectionStream */
    /** @copydoc GPUProjectionResourceOperations::createProjectionStream */
    void *AMDDeviceContext::createProjectionStream()
    {
        hipStream_t stream = nullptr;
        const auto status = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        if (status != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] createProjectionStream/hipStreamCreateWithFlags"
                      << " device=" << device_ordinal_ << " handle=" << static_cast<void *>(stream)
                      << " status=" << static_cast<int>(status) << " error=" << hipGetErrorString(status));
            return nullptr;
        }
        return static_cast<void *>(stream);
    }
    /** @copydoc GPUProjectionResourceOperations::createProjectionEvent */
    void *AMDDeviceContext::createProjectionEvent()
    {
        hipEvent_t event = nullptr;
        const auto status = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        if (status != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] createProjectionEvent/hipEventCreateWithFlags"
                      << " device=" << device_ordinal_ << " handle=" << static_cast<void *>(event)
                      << " status=" << static_cast<int>(status) << " error=" << hipGetErrorString(status));
            return nullptr;
        }
        return static_cast<void *>(event);
    }
    /** @copydoc GPUProjectionResourceOperations::destroyProjectionStream */
    bool AMDDeviceContext::destroyProjectionStream(void *stream) noexcept
    {
        const auto selected = hipSetDevice(device_ordinal_);
        if (selected != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] destroyProjectionStream/hipSetDevice"
                      << " device=" << device_ordinal_ << " handle=" << stream
                      << " status=" << static_cast<int>(selected) << " error=" << hipGetErrorString(selected));
            return false;
        }
        const auto destroyed = hipStreamDestroy(static_cast<hipStream_t>(stream));
        if (destroyed != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] destroyProjectionStream/hipStreamDestroy"
                      << " device=" << device_ordinal_ << " handle=" << stream
                      << " status=" << static_cast<int>(destroyed) << " error=" << hipGetErrorString(destroyed));
            return false;
        }
        return true;
    }
    /** @copydoc GPUProjectionResourceOperations::destroyProjectionEvent */
    bool AMDDeviceContext::destroyProjectionEvent(void *event) noexcept
    {
        const auto selected = hipSetDevice(device_ordinal_);
        if (selected != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] destroyProjectionEvent/hipSetDevice"
                      << " device=" << device_ordinal_ << " handle=" << event
                      << " status=" << static_cast<int>(selected) << " error=" << hipGetErrorString(selected));
            return false;
        }
        const auto destroyed = hipEventDestroy(static_cast<hipEvent_t>(event));
        if (destroyed != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] destroyProjectionEvent/hipEventDestroy"
                      << " device=" << device_ordinal_ << " handle=" << event
                      << " status=" << static_cast<int>(destroyed) << " error=" << hipGetErrorString(destroyed));
            return false;
        }
        return true;
    }

    void AMDDeviceContext::resetAuxiliaryStreams()
    {
        // Completed graph objects retain fixed projection handles for replay.
        // Only context cleanup retires them; this public reset clears named
        // auxiliary streams under the same exclusion as native recording.
        projection_resources_.withIdleOperation([&] {
            std::lock_guard<std::mutex> lock(auxiliary_streams_mutex_);
            for (auto &[name, stream] : auxiliary_streams_)
            {
                (void)name;
                if (stream && stream != default_stream_)
                {
                    if (!setAMDDeviceForResource(device_ordinal_, "destroyAuxiliaryStream"))
                        std::terminate();
                    const auto error = hipStreamDestroy(stream);
                    if (error != hipSuccess)
                    {
                        LOG_ERROR("[AMDDeviceContext] hipStreamDestroy(auxiliary) name=" << name
                                  << " device=" << device_ordinal_ << " handle=" << static_cast<void *>(stream)
                                  << " status=" << static_cast<int>(error) << " error=" << hipGetErrorString(error));
                        std::terminate();
                    }
                }
            }
            auxiliary_streams_.clear();
            auxiliary_stream_scheduling_classes_.clear();
        });
    }

    // ============================================================================
    // Event Access (Worker-Thread-Only)
    // ============================================================================

    /** @copydoc IWorkerGPUContext::createEvent */
    void *AMDDeviceContext::createEvent(GPUEventPurpose purpose)
    {
        // Validate before touching HIP; ordinary lifecycle receipts do not
        // request the compute/timestamp work needed only by real profiling.
        const unsigned flags = gpuEventHasTiming(purpose) ? hipEventDefault : hipEventDisableTiming;
        if (!setAMDDeviceForResource(device_ordinal_, "createEvent"))
        {
            throw std::runtime_error("AMDDeviceContext::createEvent could not select device " +
                                     std::to_string(device_ordinal_));
        }

        hipEvent_t event{};
        hipError_t err = hipEventCreateWithFlags(&event, flags);
        if (err != hipSuccess)
        {
            throw std::runtime_error("AMDDeviceContext::createEvent on device " +
                std::to_string(device_ordinal_) + ": " + hipGetErrorString(err));
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
        if (!event)
            throw std::invalid_argument("AMDDeviceContext::recordEvent requires a non-null event");
        if (!stream)
            throw std::invalid_argument(
                "AMDDeviceContext::recordEvent requires the exact non-null producer stream");

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        HIP_CHECK_VOID(hipEventRecord(hip_event, hip_stream));
    }

    bool AMDDeviceContext::recordEventChecked(void *event, void *stream)
    {
        if (!event)
            throw std::invalid_argument(
                "AMDDeviceContext::recordEventChecked requires a non-null event");
        if (!stream)
            throw std::invalid_argument(
                "AMDDeviceContext::recordEventChecked requires the exact non-null producer stream");

        if (!setAMDDeviceForResource(device_ordinal_, "recordEventChecked"))
            return false;

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        hipError_t err = hipEventRecord(hip_event, hip_stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipEventRecord failed: "
                      << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    void AMDDeviceContext::waitEvent(void *event, void *stream)
    {
        if (!event)
            throw std::invalid_argument("AMDDeviceContext::waitEvent requires a non-null event");
        if (!stream)
            throw std::invalid_argument(
                "AMDDeviceContext::waitEvent requires the exact non-null consumer stream");

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        // Make the stream wait for the event (GPU-side wait, not CPU blocking)
        HIP_CHECK_VOID(hipStreamWaitEvent(hip_stream, hip_event, 0));
    }

    bool AMDDeviceContext::waitEventChecked(void *event, void *stream)
    {
        if (!event)
            throw std::invalid_argument(
                "AMDDeviceContext::waitEventChecked requires a non-null event");
        if (!stream)
            throw std::invalid_argument(
                "AMDDeviceContext::waitEventChecked requires the exact non-null consumer stream");

        if (!setAMDDeviceForResource(device_ordinal_, "waitEventChecked"))
            return false;

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        hipError_t err = hipStreamWaitEvent(hip_stream, hip_event, 0);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipStreamWaitEvent failed: "
                      << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool AMDDeviceContext::queryEventChecked(void *event, bool &ready)
    {
        ready = false;
        if (!event)
        {
            LOG_ERROR("[AMDDeviceContext] queryEventChecked requires non-null event");
            return false;
        }
        if (!setAMDDeviceForResource(device_ordinal_, "queryEventChecked"))
        {
            return false;
        }

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        hipError_t err = hipEventQuery(hip_event);
        if (err == hipSuccess)
        {
            ready = true;
            return true;
        }
        if (err == hipErrorNotReady)
        {
            return true;
        }

        LOG_ERROR("[AMDDeviceContext] hipEventQuery failed: "
                  << hipGetErrorString(err));
        return false;
    }

    GPUStreamExecutionState AMDDeviceContext::queryStreamExecutionState(
        void *stream,
        std::string_view boundary)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                "AMDDeviceContext::queryStreamExecutionState requires the exact non-null stream");
        }
        if (boundary.empty())
        {
            throw std::invalid_argument(
                "AMDDeviceContext::queryStreamExecutionState requires a lifecycle boundary");
        }

        const hipError_t set_error = hipSetDevice(device_ordinal_);
        if (set_error != hipSuccess)
        {
            std::ostringstream message;
            message << "ROCm device selection failed while observing GPU stream"
                    << " device=ROCm:" << device_ordinal_
                    << " boundary=" << boundary
                    << " error=" << hipGetErrorString(set_error);
            throw std::runtime_error(message.str());
        }

        const hipError_t status =
            hipStreamQuery(static_cast<hipStream_t>(stream));
        if (status == hipSuccess)
            return GPUStreamExecutionState::Complete;
        if (status == hipErrorNotReady)
            return GPUStreamExecutionState::Pending;

        std::ostringstream message;
        message << "Asynchronous ROCm execution failed"
                << " device=ROCm:" << device_ordinal_
                << " boundary=" << boundary
                << " error=" << hipGetErrorString(status)
                << " code=" << static_cast<int>(status);
        throw std::runtime_error(message.str());
    }

    void AMDDeviceContext::synchronizeEvent(void *event)
    {
        (void)synchronizeEventChecked(event);
    }

    bool AMDDeviceContext::synchronizeEventChecked(void *event)
    {
        if (event == nullptr)
        {
            LOG_ERROR("[AMDDeviceContext] synchronizeEventChecked called with null event");
            return false;
        }

        if (!setAMDDeviceForResource(device_ordinal_, "synchronizeEventChecked"))
        {
            return false;
        }

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        hipError_t err = hipEventSynchronize(hip_event);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipEventSynchronize failed: "
                      << hipGetErrorString(err));
            return false;
        }
        return true;
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
        (void)synchronizeChecked();
    }

    bool AMDDeviceContext::synchronizeChecked()
    {
        if (capture_active_.load(std::memory_order_acquire))
        {
            LOG_ERROR(
                "[AMDDeviceContext] synchronizeChecked is forbidden while graph "
                "capture owns the device");
            return false;
        }

        bool ok = true;
        submitAndWait([this, &ok]()
                      {
        // Re-check inside the worker because capture ownership may change
        // between submission and execution. A requested synchronization that
        // cannot be honored is a hard ordering failure, never a successful no-op.
        if (capture_active_.load(std::memory_order_acquire)) {
            LOG_ERROR("[AMDDeviceContext] synchronizeChecked raced with graph capture "
                      "(ordinal=" << device_ordinal_ << ")");
            ok = false;
            return;
        }

        if (!setAMDDeviceForResource(device_ordinal_, "synchronize"))
        {
            ok = false;
            return;
        }

        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            LOG_ERROR("[AMDDeviceContext] hipDeviceSynchronize failed: " 
                      << hipGetErrorString(err));
            ok = false;
        } });
        return ok;
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
            LOG_ERROR(
                "[AMDDeviceContext] synchronizeStreamChecked is forbidden while "
                "graph capture owns the device");
            return false;
        }

        if (!setAMDDeviceForResource(device_ordinal_, "synchronizeStreamChecked"))
            return false;

        hipStream_t hip_stream = stream ? static_cast<hipStream_t>(stream) : hipStream_t(0);
        hipError_t err = hipStreamSynchronize(hip_stream);
        if (err != hipSuccess)
        {
            if (err == hipErrorStreamCaptureUnsupported ||
                err == hipErrorStreamCaptureImplicit)
            {
                LOG_ERROR("[AMDDeviceContext] hipStreamSynchronize would violate an active graph capture "
                          "(ordinal=" << device_ordinal_
                          << "): " << hipGetErrorString(err));
                return false;
            }
            LOG_ERROR("[AMDDeviceContext] hipStreamSynchronize failed: "
                      << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool AMDDeviceContext::insertStreamDependency(
        void *dependent_stream,
        void *dependency_stream)
    {
        if (dependent_stream == dependency_stream)
            return true;

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
            return false;
        }

        /*
         * Lease one preallocated event for the host-side record/wait pair.
         * HIP binds the queued wait to the latest event record visible at the
         * hipStreamWaitEvent call, so the lease can be released immediately
         * after that call without waiting for either device stream.
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

        hipEvent_t event = stream_dependency_events_[event_index];
        if (event == nullptr)
        {
            LOG_ERROR("[AMDDeviceContext] Persistent stream dependency event "
                      << event_index << " is not initialized");
            return false;
        }

        hipError_t err = hipEventRecord(event, src_stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipEventRecord failed in "
                      "insertStreamDependency: "
                      << hipGetErrorString(err));
            return false;
        }

        err = hipStreamWaitEvent(dep_stream, event, 0);
        if (err != hipSuccess)
        {
            LOG_ERROR("[AMDDeviceContext] hipStreamWaitEvent failed in "
                      "insertStreamDependency: "
                      << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    std::unique_ptr<IGPUGraphCapture> AMDDeviceContext::createGraphCapture()
    {
        return std::make_unique<HIPGraphCapture>(
            static_cast<hipStream_t>(defaultStream()),
            device_ordinal_, *this);
    }

    std::unique_ptr<IGPUGraphCapture> AMDDeviceContext::createGraphCapture(void *stream)
    {
        return std::make_unique<HIPGraphCapture>(
            static_cast<hipStream_t>(stream),
            device_ordinal_, *this);
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
            LOG_ERROR(
                "[AMDDeviceContext] debugSynchronize is forbidden while graph "
                "capture owns the device");
            return false;
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
