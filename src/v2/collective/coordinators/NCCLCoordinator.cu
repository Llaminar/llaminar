/**
 * @file NCCLCoordinator.cu
 * @brief Implementation of dedicated coordinator thread for NCCL collective operations
 * @author David Sanftenberg
 * @date February 2026
 *
 * This file implements the NCCLCoordinator class, which provides:
 * - Dedicated worker thread for all NCCL operations
 * - Per-device CUDA streams and completion events
 * - Proper ncclGroupStart/End semantics for multi-GPU collectives
 * - Thread-safe work queue for operation submission
 *
 * All NCCL operations are serialized on the coordinator thread to ensure
 * proper threading semantics - NCCL requires that all operations on a
 * communicator happen from the same thread.
 */

#include "NCCLCoordinator.h"
#include "../../utils/Logger.h"

#include <cuda_runtime.h>

#ifdef HAVE_NCCL
#include "../backends/NCCLDynamicLoader.h"
// Use the dynamically loaded NCCL types and functions
namespace nccl = llaminar2::nccl_dynamic;
#endif

namespace llaminar2
{

    // ============================================================================
    // Helper Macros for Error Checking
    // ============================================================================

#define CUDA_CHECK(call, msg)                                                \
    do                                                                       \
    {                                                                        \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            last_error_ = std::string(msg) + ": " + cudaGetErrorString(err); \
            LOG_ERROR("[NCCLCoordinator] " << last_error_);                  \
            return false;                                                    \
        }                                                                    \
    } while (0)

// CUDA cleanup macro that silently handles "driver shutting down" errors
// These are expected during process exit when CUDA runtime shuts down before
// our cleanup code runs. We log at TRACE level for debugging if needed.
#define CUDA_CHECK_VOID(call)                                                              \
    do                                                                                     \
    {                                                                                      \
        cudaError_t err = (call);                                                          \
        if (err != cudaSuccess)                                                            \
        {                                                                                  \
            if (err == cudaErrorCudartUnloading)                                           \
            {                                                                              \
                LOG_TRACE("[NCCLCoordinator] " << #call                                    \
                                               << " skipped: CUDA runtime shutting down"); \
            }                                                                              \
            else                                                                           \
            {                                                                              \
                LOG_WARN("[NCCLCoordinator] " << #call << " failed: "                      \
                                              << cudaGetErrorString(err));                 \
            }                                                                              \
        }                                                                                  \
    } while (0)

#ifdef HAVE_NCCL
#define NCCL_CHECK(call, msg)                                                    \
    do                                                                           \
    {                                                                            \
        nccl::ncclResult_t r = (call);                                           \
        if (r != nccl::ncclSuccess)                                              \
        {                                                                        \
            last_error_ = std::string(msg) + ": " + nccl::ncclGetErrorString(r); \
            LOG_ERROR("[NCCLCoordinator] " << last_error_);                      \
            return false;                                                        \
        }                                                                        \
    } while (0)
#endif

    // ============================================================================
    // Type Conversion Helpers
    // ============================================================================

#ifdef HAVE_NCCL
    static nccl::ncclDataType_t toNcclDataType(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return nccl::ncclFloat;
        case CollectiveDataType::FLOAT16:
            return nccl::ncclHalf;
        case CollectiveDataType::BFLOAT16:
            return nccl::ncclBfloat16;
        case CollectiveDataType::INT32:
            return nccl::ncclInt32;
        case CollectiveDataType::INT8:
            return nccl::ncclInt8;
        default:
            return nccl::ncclFloat;
        }
    }

    static nccl::ncclDataType_t toNcclDataTypeInt(int dtype_int)
    {
        switch (dtype_int)
        {
        case 0: // FLOAT32
            return nccl::ncclFloat;
        case 1: // FLOAT16
            return nccl::ncclHalf;
        case 2: // BFLOAT16
            return nccl::ncclBfloat16;
        case 3: // INT32
            return nccl::ncclInt32;
        case 4: // INT8
            return nccl::ncclInt8;
        default:
            return nccl::ncclFloat;
        }
    }

    static nccl::ncclRedOp_t toNcclRedOp(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
        case CollectiveOp::REDUCE_SCATTER:
            return nccl::ncclSum;
        case CollectiveOp::ALLREDUCE_MAX:
            return nccl::ncclMax;
        case CollectiveOp::ALLREDUCE_MIN:
            return nccl::ncclMin;
        default:
            return nccl::ncclSum;
        }
    }

    static nccl::ncclRedOp_t toNcclRedOpInt(int op_int)
    {
        switch (op_int)
        {
        case 0: // SUM
            return nccl::ncclSum;
        case 1: // PROD
            return nccl::ncclProd;
        case 2: // MIN
            return nccl::ncclMin;
        case 3: // MAX
            return nccl::ncclMax;
        default:
            return nccl::ncclSum;
        }
    }

    static int toDataTypeInt(CollectiveDataType dtype)
    {
        return static_cast<int>(dtype);
    }

    static int toOpInt(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
        case CollectiveOp::REDUCE_SCATTER:
            return 0; // SUM
        case CollectiveOp::ALLREDUCE_MAX:
            return 3; // MAX
        case CollectiveOp::ALLREDUCE_MIN:
            return 2; // MIN
        default:
            return 0; // SUM
        }
    }
#endif

    // ============================================================================
    // Constructor / Destructor
    // ============================================================================

    NCCLCoordinator::NCCLCoordinator()
    {
        LOG_DEBUG("[NCCLCoordinator] Created");
    }

    NCCLCoordinator::~NCCLCoordinator()
    {
        LOG_DEBUG("[NCCLCoordinator] Destroying");
        if (initialized_.load())
        {
            shutdown();
        }
        LOG_DEBUG("[NCCLCoordinator] Destroyed");
    }

    // ============================================================================
    // Lifecycle
    // ============================================================================

    bool NCCLCoordinator::initialize(const std::vector<int> &device_ordinals)
    {
#ifdef HAVE_NCCL
        if (initialized_.load())
        {
            LOG_WARN("[NCCLCoordinator] Already initialized, shutting down first");
            shutdown();
        }

        if (device_ordinals.empty())
        {
            last_error_ = "No device ordinals provided";
            LOG_ERROR("[NCCLCoordinator] " << last_error_);
            return false;
        }

        // Ensure NCCL is loaded
        if (!nccl::isLoaded() && !nccl::load())
        {
            const char *err = nccl::getLastError();
            last_error_ = std::string("Failed to load NCCL: ") + (err ? err : "unknown error");
            LOG_ERROR("[NCCLCoordinator] " << last_error_);
            return false;
        }

        // Store device ordinals
        device_ordinals_ = device_ordinals;
        num_devices_ = static_cast<int>(device_ordinals.size());

        LOG_DEBUG("[NCCLCoordinator] Initializing with " << num_devices_ << " devices: "
                                                         << [&]()
                  {
                  std::string s;
                  for (int i = 0; i < num_devices_; ++i)
                  {
                      if (i > 0) s += ", ";
                      s += std::to_string(device_ordinals_[i]);
                  }
                  return s; }());

        // Reset state
        init_success_.store(false);
        init_complete_.store(false);
        running_.store(false);

        // Start coordinator thread
        coordinator_thread_ = std::thread(&NCCLCoordinator::coordinatorLoop, this);

        // Wait for initialization to complete
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]()
                           { return init_complete_.load(); });
        }

        if (!init_success_.load())
        {
            // Initialization failed - join thread
            if (coordinator_thread_.joinable())
            {
                coordinator_thread_.join();
            }
            LOG_ERROR("[NCCLCoordinator] Initialization failed: " << last_error_);
            return false;
        }

        initialized_.store(true);
        LOG_INFO("[NCCLCoordinator] Initialized with " << num_devices_ << " GPU(s)");
        return true;
#else
        last_error_ = "NCCL not available (HAVE_NCCL not defined)";
        LOG_ERROR("[NCCLCoordinator] " << last_error_);
        return false;
#endif
    }

    void NCCLCoordinator::shutdown()
    {
        if (!initialized_.load() && !running_.load())
        {
            return;
        }

        LOG_DEBUG("[NCCLCoordinator] Shutting down");

        // Signal coordinator thread to stop
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            running_.store(false);
        }
        queue_cv_.notify_all();

        // Wait for coordinator thread to finish
        if (coordinator_thread_.joinable())
        {
            coordinator_thread_.join();
        }

        initialized_.store(false);
        LOG_INFO("[NCCLCoordinator] Shutdown complete");
    }

    void NCCLCoordinator::abortCommunicators()
    {
#ifdef HAVE_NCCL
        LOG_WARN("[NCCLCoordinator] Aborting all " << comms_.size()
                                                   << " communicators to unblock pending NCCL operations");

        for (int i = 0; i < static_cast<int>(comms_.size()); ++i)
        {
            if (comms_[i] != nullptr)
            {
                if (i < static_cast<int>(device_ordinals_.size()))
                {
                    CUDA_CHECK_VOID(cudaSetDevice(device_ordinals_[i]));
                }
                nccl::ncclResult_t r = nccl::ncclCommAbort(
                    static_cast<nccl::ncclComm_t>(comms_[i]));
                if (r != nccl::ncclSuccess)
                {
                    LOG_WARN("[NCCLCoordinator] ncclCommAbort on device "
                             << i << " returned: " << nccl::ncclGetErrorString(r));
                }
                comms_[i] = nullptr; // Prevent double-free in cleanup
            }
        }

        // Mark as uninitialized so no further collectives are attempted
        initialized_.store(false);

        // Signal coordinator thread to stop (it may be waiting)
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            running_.store(false);
        }
        queue_cv_.notify_all();

        LOG_WARN("[NCCLCoordinator] All communicators aborted");
#else
        LOG_WARN("[NCCLCoordinator] abortCommunicators() called but NCCL not available");
#endif
    }

    // ============================================================================
    // Synchronization with Device Workers
    // ============================================================================

    void *NCCLCoordinator::getCompletionEvent(int device_idx) const
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(completion_events_.size()))
        {
            LOG_ERROR("[NCCLCoordinator] Invalid device_idx " << device_idx);
            return nullptr;
        }
        return completion_events_[device_idx];
    }

    void NCCLCoordinator::waitForDeviceEvent(int device_idx, void *worker_event)
    {
#ifdef HAVE_NCCL
        if (device_idx < 0 || device_idx >= static_cast<int>(streams_.size()))
        {
            LOG_ERROR("[NCCLCoordinator] Invalid device_idx " << device_idx);
            return;
        }

        if (worker_event == nullptr)
        {
            return;
        }

        cudaStream_t stream = static_cast<cudaStream_t>(streams_[device_idx]);
        cudaEvent_t event = static_cast<cudaEvent_t>(worker_event);

        CUDA_CHECK_VOID(cudaStreamWaitEvent(stream, event, 0));
#endif
    }

    void NCCLCoordinator::setComputeStreams(const std::vector<void *> &compute_streams)
    {
#ifdef HAVE_NCCL
        if (static_cast<int>(compute_streams.size()) != num_devices_)
        {
            LOG_ERROR("[NCCLCoordinator] setComputeStreams: expected " << num_devices_
                                                                       << " streams, got " << compute_streams.size());
            return;
        }

        compute_streams_ = compute_streams;

        // Pre-create events for stream-level pre-sync
        // Destroy any existing events first
        for (auto &evt : compute_events_)
        {
            if (evt)
                CUDA_CHECK_VOID(cudaEventDestroy(static_cast<cudaEvent_t>(evt)));
        }
        compute_events_.resize(num_devices_);

        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[NCCLCoordinator] setComputeStreams: cudaSetDevice failed for device "
                          << device_ordinals_[i] << ": " << cudaGetErrorString(err));
                compute_streams_.clear();
                compute_events_.clear();
                return;
            }

            cudaEvent_t evt;
            err = cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[NCCLCoordinator] setComputeStreams: cudaEventCreate failed for device "
                          << device_ordinals_[i] << ": " << cudaGetErrorString(err));
                compute_streams_.clear();
                compute_events_.clear();
                return;
            }
            compute_events_[i] = static_cast<void *>(evt);
        }

        LOG_INFO("[NCCLCoordinator] Registered " << num_devices_ << " compute streams for stream-level pre-sync");
#endif
    }

    // ============================================================================
    // Work Queue Implementation
    // ============================================================================

    void NCCLCoordinator::enqueueWork(std::function<void()> work)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            work_queue_.push(std::move(work));
        }
        queue_cv_.notify_one();
    }

    // ============================================================================
    // Coordinator Thread Implementation
    // ============================================================================

    void NCCLCoordinator::coordinatorLoop()
    {
        LOG_TRACE("[NCCLCoordinator] Coordinator thread starting");

        // Initialize on this thread
        initializeOnThread();

        // Signal that initialization is complete
        init_complete_.store(true);
        queue_cv_.notify_all();

        if (!init_success_.load())
        {
            LOG_ERROR("[NCCLCoordinator] Initialization failed on coordinator thread");
            return;
        }

        running_.store(true);
        LOG_TRACE("[NCCLCoordinator] Coordinator loop started");

        // Main work loop
        while (true)
        {
            std::function<void()> work;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // Wait for work or shutdown
                queue_cv_.wait(lock, [this]()
                               { return !running_.load() || !work_queue_.empty(); });

                // Check for shutdown (drain queue first)
                if (!running_.load() && work_queue_.empty())
                {
                    break;
                }

                // Dequeue work
                if (!work_queue_.empty())
                {
                    work = std::move(work_queue_.front());
                    work_queue_.pop();
                }
            }

            // Execute work outside of lock
            if (work)
            {
                try
                {
                    work();
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[NCCLCoordinator] Exception in work: " << e.what());
                }
                catch (...)
                {
                    LOG_ERROR("[NCCLCoordinator] Unknown exception in work");
                }
            }
        }

        // Cleanup before exiting
        cleanupOnThread();
        LOG_TRACE("[NCCLCoordinator] Coordinator loop exited");
    }

    void NCCLCoordinator::initializeOnThread()
    {
#ifdef HAVE_NCCL
        LOG_TRACE("[NCCLCoordinator] Initializing NCCL resources on coordinator thread");

        // Resize vectors
        comms_.resize(num_devices_, nullptr);
        streams_.resize(num_devices_, nullptr);
        completion_events_.resize(num_devices_, nullptr);

        // Step 1: Create per-device streams and events
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              cudaGetErrorString(err);
                LOG_ERROR("[NCCLCoordinator] " << last_error_);
                cleanupOnThread();
                return;
            }

            // Create stream (non-blocking for better concurrency)
            cudaStream_t stream;
            err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaStreamCreate failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              cudaGetErrorString(err);
                LOG_ERROR("[NCCLCoordinator] " << last_error_);
                cleanupOnThread();
                return;
            }
            streams_[i] = static_cast<void *>(stream);

            // Create completion event (disable timing for performance)
            cudaEvent_t event;
            err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaEventCreate failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              cudaGetErrorString(err);
                LOG_ERROR("[NCCLCoordinator] " << last_error_);
                cleanupOnThread();
                return;
            }
            completion_events_[i] = static_cast<void *>(event);

            LOG_TRACE("[NCCLCoordinator] Created stream and event for device " << device_ordinals_[i]);
        }

        // Step 2: Generate NCCL unique ID
        // Step 2: Initialize NCCL communicators using ncclCommInitAll.
        // This is the correct API for single-process multi-GPU initialization.
        // The GroupStart/CommInitRank/GroupEnd pattern has a shared memory race
        // condition in NCCL's topology exchange layer that can cause failures
        // with multiple GPUs (threads race to attach to /dev/shm segments).
        std::vector<nccl::ncclComm_t> nccl_comms(num_devices_);
        nccl::ncclResult_t r = nccl::ncclCommInitAll(nccl_comms.data(), num_devices_, device_ordinals_.data());
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclCommInitAll failed: ") + nccl::ncclGetErrorString(r);
            LOG_ERROR("[NCCLCoordinator] " << last_error_);
            cleanupOnThread();
            return;
        }
        for (int i = 0; i < num_devices_; ++i)
        {
            comms_[i] = static_cast<void *>(nccl_comms[i]);
            LOG_TRACE("[NCCLCoordinator] Initialized NCCL comm for device " << device_ordinals_[i]);
        }

        init_success_.store(true);
        LOG_DEBUG("[NCCLCoordinator] NCCL resources initialized on coordinator thread");
#else
        last_error_ = "NCCL not available (HAVE_NCCL not defined)";
        LOG_ERROR("[NCCLCoordinator] " << last_error_);
#endif
    }

    void NCCLCoordinator::cleanupOnThread()
    {
#ifdef HAVE_NCCL
        LOG_TRACE("[NCCLCoordinator] Cleaning up NCCL resources on coordinator thread");

        // Destroy NCCL communicators
        for (int i = 0; i < static_cast<int>(comms_.size()); ++i)
        {
            if (comms_[i] != nullptr)
            {
                if (i < static_cast<int>(device_ordinals_.size()))
                {
                    CUDA_CHECK_VOID(cudaSetDevice(device_ordinals_[i]));
                }
                nccl::ncclCommDestroy(static_cast<nccl::ncclComm_t>(comms_[i]));
                comms_[i] = nullptr;
            }
        }
        comms_.clear();

        // Destroy completion events
        for (int i = 0; i < static_cast<int>(completion_events_.size()); ++i)
        {
            if (completion_events_[i] != nullptr)
            {
                if (i < static_cast<int>(device_ordinals_.size()))
                {
                    CUDA_CHECK_VOID(cudaSetDevice(device_ordinals_[i]));
                }
                CUDA_CHECK_VOID(cudaEventDestroy(static_cast<cudaEvent_t>(completion_events_[i])));
                completion_events_[i] = nullptr;
            }
        }
        completion_events_.clear();

        // Destroy streams
        for (int i = 0; i < static_cast<int>(streams_.size()); ++i)
        {
            if (streams_[i] != nullptr)
            {
                if (i < static_cast<int>(device_ordinals_.size()))
                {
                    CUDA_CHECK_VOID(cudaSetDevice(device_ordinals_[i]));
                }
                CUDA_CHECK_VOID(cudaStreamDestroy(static_cast<cudaStream_t>(streams_[i])));
                streams_[i] = nullptr;
            }
        }
        streams_.clear();

        LOG_TRACE("[NCCLCoordinator] Cleanup complete");
#endif
    }

    // ============================================================================
    // Collective Operations - Public API (thread-safe, queued)
    // ============================================================================

    bool NCCLCoordinator::allreduceMulti(const std::vector<void *> &buffers, size_t count,
                                         CollectiveDataType dtype, CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match device count (" + std::to_string(num_devices_) + ")";
            return false;
        }

        return submitAndWait([&]()
                             { return doAllreduceMulti(buffers, count, toDataTypeInt(dtype), toOpInt(op)); });
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::allreduceMultiAndSynchronize(const std::vector<void *> &buffers, size_t count,
                                                       CollectiveDataType dtype, CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match device count (" + std::to_string(num_devices_) + ")";
            return false;
        }

        return submitAndWait([&]()
                             {
            if (!doAllreduceMulti(buffers, count, toDataTypeInt(dtype), toOpInt(op)))
            {
                return false;
            }
            return doSynchronizeAll(); });
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::allreduceMultiWithComputeDeps(const std::vector<void *> &buffers, size_t count,
                                                        CollectiveDataType dtype, CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match device count (" + std::to_string(num_devices_) + ")";
            return false;
        }

        // If compute streams aren't registered, fall back to synchronous path
        if (compute_streams_.empty() ||
            static_cast<int>(compute_streams_.size()) != num_devices_)
        {
            LOG_DEBUG("[NCCLCoordinator] allreduceMultiWithComputeDeps: no compute streams, "
                      "falling back to synchronous allreduceMultiAndSynchronize");
            return allreduceMultiAndSynchronize(buffers, count, dtype, op);
        }

        // Direct execution on caller thread — bypasses submitAndWait coordinator
        // thread roundtrip (~25-35µs savings per call). Safe because:
        // 1. LocalTPContext barrier ensures only ONE thread calls this at a time
        // 2. Coordinator thread is sleeping (no queued work during inference)
        // 3. direct_exec_mutex_ prevents any rare concurrent coordinator access
        {
            std::lock_guard<std::mutex> lock(direct_exec_mutex_);
            if (!doAllreduceMulti(buffers, count, toDataTypeInt(dtype), toOpInt(op)))
            {
                return false;
            }
            return doInsertComputeStreamDeps();
        }
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::allreduceSingleDeviceAsync(void *buffer, size_t count,
                                                     CollectiveDataType dtype, CollectiveOp op,
                                                     int device_idx)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (device_idx < 0 || device_idx >= num_devices_)
        {
            last_error_ = "Invalid device_idx " + std::to_string(device_idx) +
                          " (num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        if (!buffer)
        {
            last_error_ = "Null buffer for device " + std::to_string(device_idx);
            return false;
        }

        if (compute_streams_.empty() ||
            static_cast<int>(compute_streams_.size()) != num_devices_)
        {
            last_error_ = "Compute streams not registered";
            return false;
        }

        const int ordinal = device_ordinals_[device_idx];
        cudaStream_t compute_stream = static_cast<cudaStream_t>(compute_streams_[device_idx]);
        cudaEvent_t compute_event = static_cast<cudaEvent_t>(compute_events_[device_idx]);
        cudaStream_t nccl_stream = static_cast<cudaStream_t>(streams_[device_idx]);
        cudaEvent_t completion_event = static_cast<cudaEvent_t>(completion_events_[device_idx]);
        nccl::ncclComm_t comm = static_cast<nccl::ncclComm_t>(comms_[device_idx]);

        cudaError_t err;

        err = cudaSetDevice(ordinal);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
            return false;
        }

        // Pre-sync: compute → NCCL
        err = cudaEventRecord(compute_event, compute_stream);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaEventRecord(compute) failed: ") + cudaGetErrorString(err);
            return false;
        }

        err = cudaStreamWaitEvent(nccl_stream, compute_event, 0);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaStreamWaitEvent(nccl←compute) failed: ") + cudaGetErrorString(err);
            return false;
        }

        // Non-grouped allreduce — NCCL matches calls internally
        nccl::ncclResult_t r = nccl::ncclAllReduce(
            buffer, buffer, count,
            toNcclDataTypeInt(toDataTypeInt(dtype)), toNcclRedOpInt(toOpInt(op)),
            comm, nccl_stream);
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclAllReduce failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Post-sync: NCCL → compute
        err = cudaEventRecord(completion_event, nccl_stream);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaEventRecord(completion) failed: ") + cudaGetErrorString(err);
            return false;
        }

        err = cudaStreamWaitEvent(compute_stream, completion_event, 0);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaStreamWaitEvent(compute←nccl) failed: ") + cudaGetErrorString(err);
            return false;
        }

        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::allreduceSingleDeviceOnStream(void *buffer, size_t count,
                                                        CollectiveDataType dtype, CollectiveOp op,
                                                        int device_idx, void *stream)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (device_idx < 0 || device_idx >= num_devices_)
        {
            last_error_ = "Invalid device_idx " + std::to_string(device_idx) +
                          " (num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        if (!buffer)
        {
            last_error_ = "Null buffer for device " + std::to_string(device_idx);
            return false;
        }

        if (!stream)
        {
            last_error_ = "Null stream for device " + std::to_string(device_idx);
            return false;
        }

        // Per-device resources (no locking needed — each device_idx is accessed
        // by exactly one thread in the barrier-free TP path)
        const int ordinal = device_ordinals_[device_idx];
        nccl::ncclComm_t comm = static_cast<nccl::ncclComm_t>(comms_[device_idx]);
        cudaStream_t caller_stream = static_cast<cudaStream_t>(stream);

        // Fast path: skip cudaSetDevice if this thread already has the right device.
        // In LOCAL TP decode, each worker thread always targets the same device,
        // so after the first call the branch is always taken (~1-3μs saved per call).
        static thread_local int tl_last_cuda_device = -1;
        if (tl_last_cuda_device != ordinal)
        {
            cudaError_t err = cudaSetDevice(ordinal);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                return false;
            }
            tl_last_cuda_device = ordinal;
        }

        // 2. Launch allreduce directly on the caller's stream.
        //    No cross-stream event sync needed — the caller's stream provides
        //    ordering (prior compute → allreduce → subsequent compute).
        nccl::ncclResult_t r = nccl::ncclAllReduce(
            buffer, buffer, count,
            toNcclDataTypeInt(toDataTypeInt(dtype)), toNcclRedOpInt(toOpInt(op)),
            comm, caller_stream);
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclAllReduce(on-stream) failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        return true;
#else
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)op;
        (void)device_idx;
        (void)stream;
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::allgatherMulti(const std::vector<const void *> &send_buffers,
                                         const std::vector<void *> &recv_buffers,
                                         size_t send_count, CollectiveDataType dtype)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (send_buffers.size() != static_cast<size_t>(num_devices_) ||
            recv_buffers.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer count doesn't match device count";
            return false;
        }

        return submitAndWait([&]()
                             { return doAllgatherMulti(send_buffers, recv_buffers, send_count, toDataTypeInt(dtype)); });
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::broadcastMulti(const std::vector<void *> &buffers, size_t count,
                                         CollectiveDataType dtype, int root)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer count doesn't match device count";
            return false;
        }

        if (root < 0 || root >= num_devices_)
        {
            last_error_ = "Invalid root device: " + std::to_string(root);
            return false;
        }

        return submitAndWait([&]()
                             { return doBroadcastMulti(buffers, count, toDataTypeInt(dtype), root); });
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::reduceScatterMulti(const std::vector<const void *> &send_buffers,
                                             const std::vector<void *> &recv_buffers,
                                             size_t recv_count, CollectiveDataType dtype,
                                             CollectiveOp op)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (send_buffers.size() != static_cast<size_t>(num_devices_) ||
            recv_buffers.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer count doesn't match device count";
            return false;
        }

        return submitAndWait([&]()
                             { return doReduceScatterMulti(send_buffers, recv_buffers, recv_count,
                                                           toDataTypeInt(dtype), toOpInt(op)); });
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::allreduce(void *buffer, size_t count, CollectiveDataType dtype,
                                    CollectiveOp op, int device_idx)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (device_idx < 0 || device_idx >= num_devices_)
        {
            last_error_ = "Invalid device_idx: " + std::to_string(device_idx);
            return false;
        }

        return submitAndWait([&]()
                             {
        cudaError_t err = cudaSetDevice(device_ordinals_[device_idx]);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
            return false;
        }

        nccl::ncclComm_t comm = static_cast<nccl::ncclComm_t>(comms_[device_idx]);
        cudaStream_t stream = static_cast<cudaStream_t>(streams_[device_idx]);

        nccl::ncclResult_t r = nccl::ncclAllReduce(buffer, buffer, count,
                                                   toNcclDataType(dtype), toNcclRedOp(op),
                                                   comm, stream);
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclAllReduce failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion event
        err = cudaEventRecord(static_cast<cudaEvent_t>(completion_events_[device_idx]), stream);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaEventRecord failed: ") + cudaGetErrorString(err);
            return false;
        }

        return true; });
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::synchronize()
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        return submitAndWait([&]()
                             { return doSynchronizeAll(); });
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::doSynchronizeAll()
    {
#ifdef HAVE_NCCL
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                return false;
            }

            // Stream-level sync on the NCCL stream only.
            // NCCL guarantees that when the user-provided stream completes,
            // all internal NCCL work is also complete.
            cudaStream_t nccl_stream = static_cast<cudaStream_t>(streams_[i]);
            err = cudaStreamSynchronize(nccl_stream);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaStreamSynchronize failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                return false;
            }
        }
        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::doInsertComputeStreamDeps()
    {
#ifdef HAVE_NCCL
        // Insert GPU-side dependencies: make each compute stream wait for the
        // NCCL completion event. This ensures the compute stream cannot execute
        // post-allreduce kernels until NCCL has finished, WITHOUT blocking the
        // host thread. The host returns immediately after these API calls.
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed in doInsertComputeStreamDeps: ") +
                              cudaGetErrorString(err);
                return false;
            }

            cudaStream_t compute_stream = static_cast<cudaStream_t>(compute_streams_[i]);
            cudaEvent_t completion_event = static_cast<cudaEvent_t>(completion_events_[i]);

            err = cudaStreamWaitEvent(compute_stream, completion_event, 0);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaStreamWaitEvent(compute, nccl_completion) failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[NCCLCoordinator] Inserted compute stream deps for " << num_devices_ << " devices");
        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::copy(void *dst_ptr, int dst_device_idx,
                               const void *src_ptr, int src_device_idx,
                               size_t bytes)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (bytes == 0)
        {
            return true; // No-op for zero bytes
        }

        if (!dst_ptr || !src_ptr)
        {
            last_error_ = "NCCLCoordinator::copy: null buffer pointer";
            return false;
        }

        if (dst_device_idx < 0 || dst_device_idx >= num_devices_ ||
            src_device_idx < 0 || src_device_idx >= num_devices_)
        {
            last_error_ = "NCCLCoordinator::copy: device index out of range (src=" +
                          std::to_string(src_device_idx) + " dst=" + std::to_string(dst_device_idx) +
                          " num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        // Same device - use cudaMemcpyAsync on coordinator stream (synchronous wait)
        if (src_device_idx == dst_device_idx)
        {
            return submitAndWait([&]()
                                 {
            cudaError_t err = cudaSetDevice(device_ordinals_[src_device_idx]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                return false;
            }

            cudaStream_t stream = static_cast<cudaStream_t>(streams_[src_device_idx]);
            err = cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(err);
                return false;
            }
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(err);
                return false;
            }
            return true; });
        }

        // Different devices - use NCCL send/recv with synchronization
        return submitAndWait([&]()
                             { return doCopy(dst_ptr, dst_device_idx, src_ptr, src_device_idx, bytes, /*wait_for_completion=*/true); });
#else
        (void)dst_ptr;
        (void)dst_device_idx;
        (void)src_ptr;
        (void)src_device_idx;
        (void)bytes;
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::copyAsync(void *dst_ptr, int dst_device_idx,
                                    const void *src_ptr, int src_device_idx,
                                    size_t bytes)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load())
        {
            last_error_ = "NCCLCoordinator not initialized";
            return false;
        }

        if (bytes == 0)
        {
            return true; // No-op for zero bytes
        }

        if (!dst_ptr || !src_ptr)
        {
            last_error_ = "NCCLCoordinator::copyAsync: null buffer pointer";
            return false;
        }

        if (dst_device_idx < 0 || dst_device_idx >= num_devices_ ||
            src_device_idx < 0 || src_device_idx >= num_devices_)
        {
            last_error_ = "NCCLCoordinator::copyAsync: device index out of range (src=" +
                          std::to_string(src_device_idx) + " dst=" + std::to_string(dst_device_idx) +
                          " num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        // Same device - use cudaMemcpyAsync on coordinator thread
        if (src_device_idx == dst_device_idx)
        {
            return submitAndWait([&]()
                                 {
            cudaError_t err = cudaSetDevice(device_ordinals_[src_device_idx]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                return false;
            }

            cudaStream_t stream = static_cast<cudaStream_t>(streams_[src_device_idx]);
            err = cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(err);
                return false;
            }

            // Record completion event
            err = cudaEventRecord(static_cast<cudaEvent_t>(completion_events_[src_device_idx]), stream);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaEventRecord failed: ") + cudaGetErrorString(err);
                return false;
            }
            return true; });
        }

        // Different devices - use NCCL send/recv without synchronization (async)
        return submitAndWait([&]()
                             { return doCopy(dst_ptr, dst_device_idx, src_ptr, src_device_idx, bytes, /*wait_for_completion=*/false); });
#else
        (void)dst_ptr;
        (void)dst_device_idx;
        (void)src_ptr;
        (void)src_device_idx;
        (void)bytes;
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    // ============================================================================
    // Internal Collective Implementations (called ON coordinator thread)
    // ============================================================================

    bool NCCLCoordinator::doAllreduceMulti(const std::vector<void *> &buffers, size_t count,
                                           int dtype_int, int op_int)
    {
#ifdef HAVE_NCCL
        // Pre-collective sync: ensure compute kernels have finished writing to
        // the buffers before NCCL reads them.
        // Uses stream-wait-event if compute streams registered, else device sync.
        const bool use_stream_sync = !compute_streams_.empty() &&
                                     static_cast<int>(compute_streams_.size()) == num_devices_;
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed during pre-allreduce sync: ") + cudaGetErrorString(err);
                return false;
            }
            if (use_stream_sync)
            {
                cudaStream_t compute_stream = static_cast<cudaStream_t>(compute_streams_[i]);
                cudaEvent_t compute_event = static_cast<cudaEvent_t>(compute_events_[i]);
                cudaStream_t nccl_stream = static_cast<cudaStream_t>(streams_[i]);
                err = cudaEventRecord(compute_event, compute_stream);
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaEventRecord failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
                err = cudaStreamWaitEvent(nccl_stream, compute_event, 0);
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaStreamWaitEvent failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
            }
            else
            {
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaDeviceSynchronize failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
            }
        }

        // Start NCCL group for multi-GPU operation
        nccl::ncclResult_t r = nccl::ncclGroupStart();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupStart failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Issue allreduce for each device
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                nccl::ncclGroupEnd();
                return false;
            }

            nccl::ncclComm_t comm = static_cast<nccl::ncclComm_t>(comms_[i]);
            cudaStream_t stream = static_cast<cudaStream_t>(streams_[i]);

            r = nccl::ncclAllReduce(buffers[i], buffers[i], count,
                                    toNcclDataTypeInt(dtype_int), toNcclRedOpInt(op_int),
                                    comm, stream);
            if (r != nccl::ncclSuccess)
            {
                last_error_ = std::string("ncclAllReduce failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + nccl::ncclGetErrorString(r);
                nccl::ncclGroupEnd();
                return false;
            }
        }

        // End NCCL group
        r = nccl::ncclGroupEnd();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupEnd failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion events for all devices
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice for event record failed: ") + cudaGetErrorString(err);
                return false;
            }

            err = cudaEventRecord(static_cast<cudaEvent_t>(completion_events_[i]),
                                  static_cast<cudaStream_t>(streams_[i]));
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaEventRecord failed: ") + cudaGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[NCCLCoordinator] AllreduceMulti completed: " << count << " elements");
        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::doAllgatherMulti(const std::vector<const void *> &send_buffers,
                                           const std::vector<void *> &recv_buffers,
                                           size_t send_count, int dtype_int)
    {
#ifdef HAVE_NCCL
        // Pre-collective sync: ensure compute done before NCCL reads.
        const bool use_stream_sync = !compute_streams_.empty() &&
                                     static_cast<int>(compute_streams_.size()) == num_devices_;
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed during pre-allgather sync: ") + cudaGetErrorString(err);
                return false;
            }
            if (use_stream_sync)
            {
                cudaStream_t compute_stream = static_cast<cudaStream_t>(compute_streams_[i]);
                cudaEvent_t compute_event = static_cast<cudaEvent_t>(compute_events_[i]);
                cudaStream_t nccl_stream = static_cast<cudaStream_t>(streams_[i]);
                err = cudaEventRecord(compute_event, compute_stream);
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaEventRecord failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
                err = cudaStreamWaitEvent(nccl_stream, compute_event, 0);
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaStreamWaitEvent failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
            }
            else
            {
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaDeviceSynchronize failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
            }
        }

        // Start NCCL group
        nccl::ncclResult_t r = nccl::ncclGroupStart();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupStart failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Issue allgather for each device
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                nccl::ncclGroupEnd();
                return false;
            }

            nccl::ncclComm_t comm = static_cast<nccl::ncclComm_t>(comms_[i]);
            cudaStream_t stream = static_cast<cudaStream_t>(streams_[i]);

            r = nccl::ncclAllGather(send_buffers[i], recv_buffers[i], send_count,
                                    toNcclDataTypeInt(dtype_int), comm, stream);
            if (r != nccl::ncclSuccess)
            {
                last_error_ = std::string("ncclAllGather failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + nccl::ncclGetErrorString(r);
                nccl::ncclGroupEnd();
                return false;
            }
        }

        // End NCCL group
        r = nccl::ncclGroupEnd();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupEnd failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion events
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice for event record failed: ") + cudaGetErrorString(err);
                return false;
            }

            err = cudaEventRecord(static_cast<cudaEvent_t>(completion_events_[i]),
                                  static_cast<cudaStream_t>(streams_[i]));
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaEventRecord failed: ") + cudaGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[NCCLCoordinator] AllgatherMulti completed: " << send_count << " elements per device");
        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::doBroadcastMulti(const std::vector<void *> &buffers, size_t count,
                                           int dtype_int, int root)
    {
#ifdef HAVE_NCCL
        // Pre-collective sync: ensure compute done before NCCL reads.
        const bool use_stream_sync = !compute_streams_.empty() &&
                                     static_cast<int>(compute_streams_.size()) == num_devices_;
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed during pre-broadcast sync: ") + cudaGetErrorString(err);
                return false;
            }
            if (use_stream_sync)
            {
                cudaStream_t compute_stream = static_cast<cudaStream_t>(compute_streams_[i]);
                cudaEvent_t compute_event = static_cast<cudaEvent_t>(compute_events_[i]);
                cudaStream_t nccl_stream = static_cast<cudaStream_t>(streams_[i]);
                err = cudaEventRecord(compute_event, compute_stream);
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaEventRecord failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
                err = cudaStreamWaitEvent(nccl_stream, compute_event, 0);
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaStreamWaitEvent failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
            }
            else
            {
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaDeviceSynchronize failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
            }
        }

        // Start NCCL group
        nccl::ncclResult_t r = nccl::ncclGroupStart();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupStart failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Issue broadcast for each device
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                nccl::ncclGroupEnd();
                return false;
            }

            nccl::ncclComm_t comm = static_cast<nccl::ncclComm_t>(comms_[i]);
            cudaStream_t stream = static_cast<cudaStream_t>(streams_[i]);

            // ncclBroadcast: sendbuff and recvbuff can be the same for in-place
            r = nccl::ncclBroadcast(buffers[i], buffers[i], count,
                                    toNcclDataTypeInt(dtype_int), root, comm, stream);
            if (r != nccl::ncclSuccess)
            {
                last_error_ = std::string("ncclBroadcast failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + nccl::ncclGetErrorString(r);
                nccl::ncclGroupEnd();
                return false;
            }
        }

        // End NCCL group
        r = nccl::ncclGroupEnd();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupEnd failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion events
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice for event record failed: ") + cudaGetErrorString(err);
                return false;
            }

            err = cudaEventRecord(static_cast<cudaEvent_t>(completion_events_[i]),
                                  static_cast<cudaStream_t>(streams_[i]));
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaEventRecord failed: ") + cudaGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[NCCLCoordinator] BroadcastMulti completed: " << count << " elements from root " << root);
        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::doReduceScatterMulti(const std::vector<const void *> &send_buffers,
                                               const std::vector<void *> &recv_buffers,
                                               size_t recv_count, int dtype_int, int op_int)
    {
#ifdef HAVE_NCCL
        // Pre-collective sync: ensure compute done before NCCL reads.
        const bool use_stream_sync = !compute_streams_.empty() &&
                                     static_cast<int>(compute_streams_.size()) == num_devices_;
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed during pre-reducescatter sync: ") + cudaGetErrorString(err);
                return false;
            }
            if (use_stream_sync)
            {
                cudaStream_t compute_stream = static_cast<cudaStream_t>(compute_streams_[i]);
                cudaEvent_t compute_event = static_cast<cudaEvent_t>(compute_events_[i]);
                cudaStream_t nccl_stream = static_cast<cudaStream_t>(streams_[i]);
                err = cudaEventRecord(compute_event, compute_stream);
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaEventRecord failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
                err = cudaStreamWaitEvent(nccl_stream, compute_event, 0);
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaStreamWaitEvent failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
            }
            else
            {
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    last_error_ = std::string("cudaDeviceSynchronize failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + cudaGetErrorString(err);
                    return false;
                }
            }
        }

        // Start NCCL group
        nccl::ncclResult_t r = nccl::ncclGroupStart();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupStart failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Issue reduce-scatter for each device
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                nccl::ncclGroupEnd();
                return false;
            }

            nccl::ncclComm_t comm = static_cast<nccl::ncclComm_t>(comms_[i]);
            cudaStream_t stream = static_cast<cudaStream_t>(streams_[i]);

            r = nccl::ncclReduceScatter(send_buffers[i], recv_buffers[i], recv_count,
                                        toNcclDataTypeInt(dtype_int), toNcclRedOpInt(op_int),
                                        comm, stream);
            if (r != nccl::ncclSuccess)
            {
                last_error_ = std::string("ncclReduceScatter failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + nccl::ncclGetErrorString(r);
                nccl::ncclGroupEnd();
                return false;
            }
        }

        // End NCCL group
        r = nccl::ncclGroupEnd();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupEnd failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion events
        for (int i = 0; i < num_devices_; ++i)
        {
            cudaError_t err = cudaSetDevice(device_ordinals_[i]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice for event record failed: ") + cudaGetErrorString(err);
                return false;
            }

            err = cudaEventRecord(static_cast<cudaEvent_t>(completion_events_[i]),
                                  static_cast<cudaStream_t>(streams_[i]));
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaEventRecord failed: ") + cudaGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[NCCLCoordinator] ReduceScatterMulti completed: " << recv_count << " elements per device");
        return true;
#else
        last_error_ = "NCCL not available";
        return false;
#endif
    }

    bool NCCLCoordinator::doCopy(void *dst_ptr, int dst_device_idx,
                                 const void *src_ptr, int src_device_idx,
                                 size_t bytes, bool wait_for_completion)
    {
#ifdef HAVE_NCCL
        // Start NCCL group for paired send/recv
        nccl::ncclResult_t r = nccl::ncclGroupStart();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupStart failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        // Issue send from source device
        cudaError_t err = cudaSetDevice(device_ordinals_[src_device_idx]);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaSetDevice (src) failed: ") + cudaGetErrorString(err);
            nccl::ncclGroupEnd();
            return false;
        }

        nccl::ncclComm_t src_comm = static_cast<nccl::ncclComm_t>(comms_[src_device_idx]);
        cudaStream_t src_stream = static_cast<cudaStream_t>(streams_[src_device_idx]);

        // ncclSend: peer rank is the destination device index within the communicator
        r = nccl::ncclSend(src_ptr, bytes, nccl::ncclInt8, dst_device_idx, src_comm, src_stream);
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclSend failed: ") + nccl::ncclGetErrorString(r);
            nccl::ncclGroupEnd();
            return false;
        }

        // Issue recv on destination device
        err = cudaSetDevice(device_ordinals_[dst_device_idx]);
        if (err != cudaSuccess)
        {
            last_error_ = std::string("cudaSetDevice (dst) failed: ") + cudaGetErrorString(err);
            nccl::ncclGroupEnd();
            return false;
        }

        nccl::ncclComm_t dst_comm = static_cast<nccl::ncclComm_t>(comms_[dst_device_idx]);
        cudaStream_t dst_stream = static_cast<cudaStream_t>(streams_[dst_device_idx]);

        // ncclRecv: peer rank is the source device index within the communicator
        r = nccl::ncclRecv(dst_ptr, bytes, nccl::ncclInt8, src_device_idx, dst_comm, dst_stream);
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclRecv failed: ") + nccl::ncclGetErrorString(r);
            nccl::ncclGroupEnd();
            return false;
        }

        // End NCCL group - this enqueues the actual transfer on the streams
        r = nccl::ncclGroupEnd();
        if (r != nccl::ncclSuccess)
        {
            last_error_ = std::string("ncclGroupEnd failed: ") + nccl::ncclGetErrorString(r);
            return false;
        }

        if (wait_for_completion)
        {
            // Synchronize both streams to ensure copy is complete
            err = cudaSetDevice(device_ordinals_[src_device_idx]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice (sync src) failed: ") + cudaGetErrorString(err);
                return false;
            }
            err = cudaStreamSynchronize(src_stream);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaStreamSynchronize (src) failed: ") + cudaGetErrorString(err);
                return false;
            }

            err = cudaSetDevice(device_ordinals_[dst_device_idx]);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice (sync dst) failed: ") + cudaGetErrorString(err);
                return false;
            }
            err = cudaStreamSynchronize(dst_stream);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaStreamSynchronize (dst) failed: ") + cudaGetErrorString(err);
                return false;
            }
        }

        // Record completion events (for both sync and async paths)
        err = cudaSetDevice(device_ordinals_[src_device_idx]);
        if (err == cudaSuccess)
        {
            CUDA_CHECK_VOID(cudaEventRecord(static_cast<cudaEvent_t>(completion_events_[src_device_idx]), src_stream));
        }
        err = cudaSetDevice(device_ordinals_[dst_device_idx]);
        if (err == cudaSuccess)
        {
            CUDA_CHECK_VOID(cudaEventRecord(static_cast<cudaEvent_t>(completion_events_[dst_device_idx]), dst_stream));
        }

        LOG_DEBUG("[NCCLCoordinator] Copy " << (wait_for_completion ? "completed" : "enqueued")
                                            << ": " << bytes << " bytes from device "
                                            << device_ordinals_[src_device_idx] << " to device " << device_ordinals_[dst_device_idx]);
        return true;
#else
        (void)dst_ptr;
        (void)dst_device_idx;
        (void)src_ptr;
        (void)src_device_idx;
        (void)bytes;
        (void)wait_for_completion;
        last_error_ = "NCCL not available";
        return false;
#endif
    }

} // namespace llaminar2
