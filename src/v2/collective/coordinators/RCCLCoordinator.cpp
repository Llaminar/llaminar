/**
 * @file RCCLCoordinator.cpp
 * @brief Implementation of dedicated coordinator thread for RCCL collective operations
 * @author David Sanftenberg
 * @date February 2026
 *
 * This file implements the RCCLCoordinator class, which provides:
 * - Dedicated worker thread for all RCCL operations
 * - Per-device HIP streams and completion events
 * - Proper rcclGroupStart/End semantics for multi-GPU collectives
 * - Thread-safe work queue for operation submission
 *
 * Setup and grouped host transactions use the coordinator worker. Captured
 * participant-local operations enqueue directly on their caller's explicit
 * stream; each participant retains a consistent communicator operation order.
 * No coordinator queue, default stream or host wait is inserted into those
 * captured operations, including output-partitioned sum/reduce-scatter.
 * Used and unused communicators share mandatory native finalization; retirement
 * performs no allocation or synthetic collective and never abandons handles.
 */

#include "RCCLCoordinator.h"
#include "../NativeReduceScatterContract.h"
#include "../NativeCollectiveRowsContract.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"

#include <algorithm>
#include <functional>

#ifdef HAVE_RCCL
#include <hip/hip_runtime.h>
#include "../backends/RCCLDynamicLoader.h"
#include "../../backends/rocm/HipDeviceGuard.h"
// Use the dynamically loaded RCCL types and functions
namespace rccl = llaminar2::rccl_dynamic;
#endif

namespace llaminar2
{

#ifdef HAVE_RCCL
    static hipError_t trackedHipSetDevice(int device_ordinal)
    {
        return static_cast<hipError_t>(HipDeviceGuard::forceSetDevice(device_ordinal));
    }
#endif

    // ============================================================================
    // Helper Macros for Error Checking
    // ============================================================================

#ifdef HAVE_RCCL
#define HIP_CHECK(call, msg)                                                \
    do                                                                      \
    {                                                                       \
        hipError_t err = (call);                                            \
        if (err != hipSuccess)                                              \
        {                                                                   \
            last_error_ = std::string(msg) + ": " + hipGetErrorString(err); \
            LOG_ERROR("[RCCLCoordinator] " << last_error_);                 \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define HIP_CHECK_VOID(call)                                          \
    do                                                                \
    {                                                                 \
        hipError_t err = (call);                                      \
        if (err != hipSuccess)                                        \
        {                                                             \
            LOG_WARN("[RCCLCoordinator] " << #call << " failed: "     \
                                          << hipGetErrorString(err)); \
        }                                                             \
    } while (0)

#define RCCL_CHECK(call, msg)                                                    \
    do                                                                           \
    {                                                                            \
        rccl::ncclResult_t r = (call);                                           \
        if (r != rccl::ncclSuccess)                                              \
        {                                                                        \
            last_error_ = std::string(msg) + ": " + rccl::ncclGetErrorString(r); \
            LOG_ERROR("[RCCLCoordinator] " << last_error_);                      \
            return false;                                                        \
        }                                                                        \
    } while (0)
#endif

    // ============================================================================
    // Type Conversion Helpers
    // ============================================================================

#ifdef HAVE_RCCL
    static rccl::ncclDataType_t toRcclDataType(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return rccl::ncclFloat;
        case CollectiveDataType::FLOAT16:
            return rccl::ncclHalf;
        case CollectiveDataType::BFLOAT16:
            return rccl::ncclBfloat16;
        case CollectiveDataType::INT32:
            return rccl::ncclInt32;
        case CollectiveDataType::INT8:
            return rccl::ncclInt8;
        default:
            return rccl::ncclFloat;
        }
    }

    static rccl::ncclDataType_t toRcclDataTypeInt(int dtype_int)
    {
        switch (dtype_int)
        {
        case 0: // FLOAT32
            return rccl::ncclFloat;
        case 1: // FLOAT16
            return rccl::ncclHalf;
        case 2: // BFLOAT16
            return rccl::ncclBfloat16;
        case 3: // INT32
            return rccl::ncclInt32;
        case 4: // INT8
            return rccl::ncclInt8;
        default:
            return rccl::ncclFloat;
        }
    }

    static rccl::ncclRedOp_t toRcclRedOp(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
        case CollectiveOp::REDUCE_SCATTER:
            return rccl::ncclSum;
        case CollectiveOp::ALLREDUCE_MAX:
            return rccl::ncclMax;
        case CollectiveOp::ALLREDUCE_MIN:
            return rccl::ncclMin;
        default:
            return rccl::ncclSum;
        }
    }

    static rccl::ncclRedOp_t toRcclRedOpInt(int op_int)
    {
        switch (op_int)
        {
        case 0: // SUM
            return rccl::ncclSum;
        case 1: // PROD
            return rccl::ncclProd;
        case 2: // MIN
            return rccl::ncclMin;
        case 3: // MAX
            return rccl::ncclMax;
        default:
            return rccl::ncclSum;
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

    RCCLCoordinator::RCCLCoordinator()
    {
        LOG_DEBUG("[RCCLCoordinator] Created");
    }

    RCCLCoordinator::~RCCLCoordinator()
    {
        LOG_DEBUG("[RCCLCoordinator] Destroying");
        if (initialized_.load() || running_.load() || coordinator_thread_.joinable())
        {
            shutdown();
        }
        LOG_DEBUG("[RCCLCoordinator] Destroyed");
    }

    // ============================================================================
    // Lifecycle
    // ============================================================================

    bool RCCLCoordinator::initialize(const std::vector<int> &device_ordinals)
    {
#ifdef HAVE_RCCL
        if (initialized_.load())
        {
            LOG_WARN("[RCCLCoordinator] Already initialized, shutting down first");
            shutdown();
        }

        if (device_ordinals.empty())
        {
            last_error_ = "No device ordinals provided";
            LOG_ERROR("[RCCLCoordinator] " << last_error_);
            return false;
        }

        // Ensure RCCL is loaded
        if (!rccl::isLoaded() && !rccl::load())
        {
            const char *err = rccl::getLastError();
            last_error_ = std::string("Failed to load RCCL: ") + (err ? err : "unknown error");
            LOG_ERROR("[RCCLCoordinator] " << last_error_);
            return false;
        }

        if (!rccl::isCommFinalizeAvailable())
        {
            last_error_ = "RCCL runtime lacks required communicator finalization";
            LOG_ERROR("[RCCLCoordinator] " << last_error_);
            return false;
        }

        // Store device ordinals
        device_ordinals_ = device_ordinals;
        num_devices_ = static_cast<int>(device_ordinals.size());

        LOG_DEBUG("[RCCLCoordinator] Initializing with " << num_devices_ << " devices: "
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
        collective_performed_.store(false, std::memory_order_release);
        init_success_.store(false);
        init_complete_.store(false);
        running_.store(false);

        // Start coordinator thread
        coordinator_thread_ = std::thread(&RCCLCoordinator::coordinatorLoop, this);

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
            LOG_ERROR("[RCCLCoordinator] Initialization failed: " << last_error_);
            return false;
        }

        initialized_.store(true);
        LOG_DEBUG("[RCCLCoordinator] Initialized with " << num_devices_ << " ROCm GPU(s)");
        return true;
#else
        last_error_ = "RCCL not available (HAVE_RCCL not defined)";
        LOG_ERROR("[RCCLCoordinator] " << last_error_);
        return false;
#endif
    }

    void RCCLCoordinator::shutdown()
    {
        if (!initialized_.load() && !running_.load() && !coordinator_thread_.joinable())
        {
            return;
        }

        LOG_DEBUG("[RCCLCoordinator] Shutting down");

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
        LOG_DEBUG("[RCCLCoordinator] Shutdown complete");
    }

    void RCCLCoordinator::abortCommunicators()
    {
#ifdef HAVE_RCCL
        std::lock_guard<std::mutex> abort_lock(abort_mutex_);

        /*
         * RCCL inherits NCCL's all-active-ranks abort contract. Calling local
         * ranks serially can strand rank zero waiting for a rank the host has
         * not entered yet. Close collective admission, detach every handle from
         * ordinary cleanup, and make all active ranks enter ncclCommAbort before
         * joining any of them. This mirrors CUDA and keeps backend failure
         * lifecycles symmetric.
         *
         * An unused communicator has no collective device work to interrupt.
         * Leave its handles to ordinary native finalization; cleanup never
         * manufactures a first collective merely to retire an unused owner.
         */
        initialized_.store(false, std::memory_order_release);
        const bool has_collective_work =
            collective_performed_.load(std::memory_order_acquire);

        std::vector<void *> communicators;
        if (has_collective_work)
        {
            communicators = comms_;
        }
        else
        {
            communicators.assign(comms_.size(), nullptr);
        }

        size_t active_count = 0;
        for (void *communicator : communicators)
            active_count += communicator != nullptr ? 1U : 0U;
        LOG_WARN("[RCCLCoordinator] Aborting " << active_count
                                                << " active communicator ranks concurrently"
                                                << " collective_performed="
                                                << has_collective_work);

        std::vector<hipError_t> device_results(
            communicators.size(), hipSuccess);
        std::vector<rccl::ncclResult_t> abort_results(
            communicators.size(), rccl::ncclSuccess);
        std::vector<std::thread> abort_threads;
        abort_threads.reserve(active_count);

        for (size_t rank = 0; rank < communicators.size(); ++rank)
        {
            if (!communicators[rank])
                continue;
            abort_threads.emplace_back([&, rank]()
            {
                if (rank >= device_ordinals_.size())
                {
                    device_results[rank] = hipErrorInvalidDevice;
                    abort_results[rank] = rccl::ncclInvalidArgument;
                    return;
                }
                device_results[rank] =
                    trackedHipSetDevice(device_ordinals_[rank]);
                if (device_results[rank] != hipSuccess)
                {
                    abort_results[rank] = rccl::ncclUnhandledCudaError;
                    return;
                }
                abort_results[rank] = rccl::ncclCommAbort(
                    static_cast<rccl::ncclComm_t>(communicators[rank]));
            });
        }

        /* Every active local rank has entered RCCL before any join occurs. */
        for (std::thread &thread : abort_threads)
            thread.join();

        if (has_collective_work)
        {
            /*
             * Preserve stable handles until abort has interrupted every caller
             * that crossed admission before the fatal flag. Cleanup begins only
             * after these entries are invalidated.
             */
            std::fill(comms_.begin(), comms_.end(), nullptr);
        }

        for (size_t rank = 0; rank < communicators.size(); ++rank)
        {
            if (!communicators[rank])
                continue;
            if (device_results[rank] != hipSuccess)
            {
                LOG_ERROR("[RCCLCoordinator] Fatal HIP device selection failure while aborting rank "
                          << rank << " device="
                          << (rank < device_ordinals_.size()
                                  ? device_ordinals_[rank]
                                  : -1)
                          << " error="
                          << hipGetErrorString(device_results[rank]));
                continue;
            }
            if (abort_results[rank] != rccl::ncclSuccess)
            {
                LOG_ERROR("[RCCLCoordinator] ncclCommAbort failed for rank "
                          << rank << " device=" << device_ordinals_[rank]
                          << " error="
                          << rccl::ncclGetErrorString(abort_results[rank]));
            }
        }

        // Signal coordinator thread to stop (it may be waiting)
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            running_.store(false);
        }
        queue_cv_.notify_all();

        LOG_WARN("[RCCLCoordinator] Concurrent communicator abort complete");
#else
        LOG_WARN("[RCCLCoordinator] abortCommunicators() called but RCCL not available");
#endif
    }

    // ============================================================================
    // Synchronization with Device Workers
    // ============================================================================

    void *RCCLCoordinator::getCompletionEvent(int device_idx) const
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(completion_events_.size()))
        {
            LOG_ERROR("[RCCLCoordinator] Invalid device_idx " << device_idx);
            return nullptr;
        }
        return completion_events_[device_idx];
    }

    void RCCLCoordinator::waitForDeviceEvent(int device_idx, void *worker_event)
    {
#ifdef HAVE_RCCL
        if (device_idx < 0 || device_idx >= static_cast<int>(streams_.size()))
        {
            LOG_ERROR("[RCCLCoordinator] Invalid device_idx " << device_idx);
            return;
        }

        if (worker_event == nullptr)
        {
            return;
        }

        hipStream_t stream = static_cast<hipStream_t>(streams_[device_idx]);
        hipEvent_t event = static_cast<hipEvent_t>(worker_event);

        if (debugEnv().validation.validate_gpu_ptrs)
        {
            int current_dev = -1;
            (void)hipGetDevice(&current_dev);
            LOG_DEBUG("[RCCL_STREAM_WAIT] slot=" << device_idx
                                                 << " target_device=" << device_ordinals_[device_idx]
                                                 << " current_device=" << current_dev
                                                 << " stream=" << stream
                                                 << " event=" << event);
        }

        HIP_CHECK_VOID(hipStreamWaitEvent(stream, event, 0));
#endif
    }

    void RCCLCoordinator::setComputeStreams(const std::vector<void *> &compute_streams)
    {
#ifdef HAVE_RCCL
        if (static_cast<int>(compute_streams.size()) != num_devices_)
        {
            LOG_ERROR("[RCCLCoordinator] setComputeStreams: expected " << num_devices_
                                                                       << " streams, got " << compute_streams.size());
            return;
        }

        compute_streams_ = compute_streams;

        // Pre-create events for recording on compute streams (one per device)
        // These are used in doAllreduceMulti to establish stream-level dependencies
        // instead of expensive hipDeviceSynchronize().
        compute_events_.resize(num_devices_, nullptr);
        for (int i = 0; i < num_devices_; ++i)
        {
            if (compute_events_[i])
            {
                continue; // Already created
            }
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                LOG_ERROR("[RCCLCoordinator] setComputeStreams: hipSetDevice failed for device "
                          << device_ordinals_[i]);
                compute_streams_.clear();
                return;
            }
            hipEvent_t ev;
            err = hipEventCreateWithFlags(&ev, hipEventDisableTiming);
            if (err != hipSuccess)
            {
                LOG_ERROR("[RCCLCoordinator] setComputeStreams: hipEventCreate failed for device "
                          << device_ordinals_[i]);
                compute_streams_.clear();
                return;
            }
            compute_events_[i] = static_cast<void *>(ev);
        }

        LOG_DEBUG("[RCCLCoordinator] Compute streams registered for " << num_devices_
                                                                     << " devices — using stream-level pre-sync");
        if (debugEnv().tp_collective_contract_trace)
        {
            for (int i = 0; i < num_devices_; ++i)
            {
                LOG_DEBUG("[TP_COLLECTIVE_CONTEXT] event=rccl_compute_stream"
                         << " coordinator=" << static_cast<const void *>(this)
                         << " slot=" << i
                         << " ordinal=" << device_ordinals_[i]
                         << " compute_stream=" << compute_streams_[i]
                         << " compute_event=" << compute_events_[i]);
            }
        }
#else
        (void)compute_streams;
#endif
    }

    // ============================================================================
    // Work Queue Implementation
    // ============================================================================

    void RCCLCoordinator::enqueueWork(std::function<void()> work)
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

    void RCCLCoordinator::coordinatorLoop()
    {
        LOG_TRACE("[RCCLCoordinator] Coordinator thread starting");

        // Initialize on this thread
        initializeOnThread();

        // Signal that initialization is complete
        init_complete_.store(true);
        queue_cv_.notify_all();

        if (!init_success_.load())
        {
            LOG_ERROR("[RCCLCoordinator] Initialization failed on coordinator thread");
            return;
        }

        running_.store(true);
        LOG_TRACE("[RCCLCoordinator] Coordinator loop started");

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
                    LOG_ERROR("[RCCLCoordinator] Exception in work: " << e.what());
                }
                catch (...)
                {
                    LOG_ERROR("[RCCLCoordinator] Unknown exception in work");
                }
            }
        }

        // Cleanup before exiting
        cleanupOnThread();
        LOG_TRACE("[RCCLCoordinator] Coordinator loop exited");
    }

    void RCCLCoordinator::initializeOnThread()
    {
#ifdef HAVE_RCCL
        LOG_TRACE("[RCCLCoordinator] Initializing RCCL resources on coordinator thread");

        // Resize vectors
        comms_.resize(num_devices_, nullptr);
        streams_.resize(num_devices_, nullptr);
        completion_events_.resize(num_devices_, nullptr);

        // Step 1: Create per-device streams and events
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              hipGetErrorString(err);
                LOG_ERROR("[RCCLCoordinator] " << last_error_);
                cleanupOnThread();
                return;
            }

            // Create stream (non-blocking for better concurrency)
            hipStream_t stream;
            err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipStreamCreate failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              hipGetErrorString(err);
                LOG_ERROR("[RCCLCoordinator] " << last_error_);
                cleanupOnThread();
                return;
            }
            streams_[i] = static_cast<void *>(stream);

            // Create completion event (disable timing for performance)
            hipEvent_t event;
            err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipEventCreate failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              hipGetErrorString(err);
                LOG_ERROR("[RCCLCoordinator] " << last_error_);
                cleanupOnThread();
                return;
            }
            completion_events_[i] = static_cast<void *>(event);

            LOG_TRACE("[RCCLCoordinator] Created stream and event for device " << device_ordinals_[i]);
        }

        // Step 2: Initialize RCCL communicators using rcclCommInitAll.
        // This is the correct API for single-process multi-GPU initialization.
        // The GroupStart/CommInitRank/GroupEnd pattern has a shared memory race
        // condition in RCCL's topology exchange layer that causes failures with
        // >=4 GPUs (threads race to attach to /dev/shm segments before they exist).
        std::vector<rccl::ncclComm_t> rccl_comms(num_devices_);
        rccl::ncclResult_t r = rccl::ncclCommInitAll(rccl_comms.data(), num_devices_, device_ordinals_.data());
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclCommInitAll failed: ") + rccl::ncclGetErrorString(r);
            LOG_ERROR("[RCCLCoordinator] " << last_error_);
            cleanupOnThread();
            return;
        }
        for (int i = 0; i < num_devices_; ++i)
        {
            comms_[i] = static_cast<void *>(rccl_comms[i]);
            LOG_TRACE("[RCCLCoordinator] Initialized RCCL comm for device " << device_ordinals_[i]);
        }

        init_success_.store(true);
        LOG_DEBUG("[RCCLCoordinator] RCCL resources initialized on coordinator thread");
#else
        last_error_ = "RCCL not available (HAVE_RCCL not defined)";
        LOG_ERROR("[RCCLCoordinator] " << last_error_);
#endif
    }

    /**
     * @brief Retire every native owner on its coordinator thread, including unused cliques.
     *
     * Teardown joins existing work, finalizes the complete clique, then destroys
     * communicators before their events/streams. It creates no device work or
     * storage. Native lifecycle failures are fatal rather than leaking handles
     * or returning a partially retired owner to the process pool.
     */
    void RCCLCoordinator::cleanupOnThread()
    {
#ifdef HAVE_RCCL
        LOG_TRACE("[RCCLCoordinator] Cleaning up RCCL resources on coordinator thread");

        const auto require_hip = [&](hipError_t status, const char *operation, std::size_t rank)
        {
            if (status == hipSuccess) return;
            LOG_ERROR("[RCCLCoordinator] Retirement " << operation << " failed for rank " << rank
                      << ": " << hipGetErrorString(status));
            std::terminate();
        };
        const auto select_device = [&](std::size_t rank)
        {
            if (rank >= device_ordinals_.size())
            {
                LOG_ERROR("[RCCLCoordinator] Retirement has a foreign endpoint rank " << rank);
                std::terminate();
            }
            require_hip(trackedHipSetDevice(device_ordinals_[rank]), "device selection", rank);
        };

        // Step 1: Synchronize all streams before destroying any resources.
        // This ensures any internally-queued RCCL/HIP work completes before we
        // free communicators, events, or streams.
        for (int i = 0; i < static_cast<int>(streams_.size()); ++i)
        {
            if (streams_[i] != nullptr)
            {
                select_device(i);
                require_hip(hipStreamSynchronize(static_cast<hipStream_t>(streams_[i])), "stream completion", i);
            }
        }

        // Unused communicators retire through the same native finalization
        // contract as used ones. Teardown must not allocate unadmitted buffers
        // or launch a synthetic collective, especially at process retirement.
        // Finalize the complete clique before releasing any communicator.
        // Finalize is a required runtime ABI, authenticated during initialize.
        for (std::size_t i = 0; i < comms_.size(); ++i)
        {
            if (!comms_[i]) continue;
            select_device(i);
            const auto status = rccl::ncclCommFinalize(static_cast<rccl::ncclComm_t>(comms_[i]));
            if (status != rccl::ncclSuccess)
            {
                LOG_ERROR("[RCCLCoordinator] Communicator finalization failed on device "
                          << device_ordinals_[i] << ": " << rccl::ncclGetErrorString(status));
                std::terminate();
            }
        }
        for (std::size_t i = 0; i < streams_.size(); ++i)
        {
            if (!streams_[i]) continue;
            select_device(i);
            require_hip(hipStreamSynchronize(static_cast<hipStream_t>(streams_[i])), "stream completion", i);
        }
        for (std::size_t i = 0; i < comms_.size(); ++i)
        {
            if (!comms_[i]) continue;
            select_device(i);
            const auto status = rccl::ncclCommDestroy(static_cast<rccl::ncclComm_t>(comms_[i]));
            if (status != rccl::ncclSuccess)
            {
                LOG_ERROR("[RCCLCoordinator] Communicator destruction failed on device "
                          << device_ordinals_[i] << ": " << rccl::ncclGetErrorString(status));
                std::terminate();
            }
            comms_[i] = nullptr;
        }
        comms_.clear();

        // Step 3: Destroy completion events
        for (int i = 0; i < static_cast<int>(completion_events_.size()); ++i)
        {
            if (completion_events_[i] != nullptr)
            {
                select_device(i);
                require_hip(hipEventDestroy(static_cast<hipEvent_t>(completion_events_[i])), "event destruction", i);
                completion_events_[i] = nullptr;
            }
        }
        completion_events_.clear();

        // Step 4: Destroy streams (after comms and events are already gone)
        for (int i = 0; i < static_cast<int>(streams_.size()); ++i)
        {
            if (streams_[i] != nullptr)
            {
                select_device(i);
                require_hip(hipStreamDestroy(static_cast<hipStream_t>(streams_[i])), "stream destruction", i);
                streams_[i] = nullptr;
            }
        }
        streams_.clear();

        LOG_TRACE("[RCCLCoordinator] Cleanup complete");
#endif
    }

    // ============================================================================
    // Collective Operations - Public API (thread-safe, queued)
    // ============================================================================

    bool RCCLCoordinator::allreduceMulti(const std::vector<void *> &buffers, size_t count,
                                         CollectiveDataType dtype, CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
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
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allreduceMultiAndSynchronize(const std::vector<void *> &buffers, size_t count,
                                                       CollectiveDataType dtype, CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
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
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allreduceMultiWithComputeDeps(const std::vector<void *> &buffers, size_t count,
                                                        CollectiveDataType dtype, CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") doesn't match device count (" + std::to_string(num_devices_) + ")";
            return false;
        }

        // Compute-stream registration is the ownership contract for this API.
        if (compute_streams_.empty() ||
            static_cast<int>(compute_streams_.size()) != num_devices_)
        {
            last_error_ =
                "allreduceMultiWithComputeDeps requires one registered compute stream per device; "
                "synchronous fallback is forbidden";
            return false;
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
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allreduceMultiOnStreams(const std::vector<void *> &buffers, size_t count,
                                                  CollectiveDataType dtype, CollectiveOp op,
                                                  const std::vector<void *> &streams)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_devices_) ||
            streams.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer/stream count does not match device count";
            return false;
        }

        for (int i = 0; i < num_devices_; ++i)
        {
            if (!buffers[i])
            {
                last_error_ = "Null allreduce buffer for device " + std::to_string(i);
                return false;
            }
            if (!streams[i])
            {
                last_error_ = "Null allreduce stream for device " + std::to_string(i);
                return false;
            }
        }

        const bool trace_device_state = debugEnv().validation.validate_gpu_ptrs;
        const size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
        std::lock_guard<std::mutex> lock(direct_exec_mutex_);

        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                rccl::ncclGroupEnd();
                return false;
            }

            if (trace_device_state)
            {
                int current_device = -1;
                hipError_t get_device = hipGetDevice(&current_device);
                if (get_device == hipSuccess)
                {
                    LOG_DEBUG("[RCCL_STREAM_GROUP_LAUNCH] thread=" << thread_hash
                                                                   << " slot=" << i
                                                                   << " target_device=" << device_ordinals_[i]
                                                                   << " current_device=" << current_device
                                                                   << " stream=" << streams[i]
                                                                   << " buffer=" << buffers[i]
                                                                   << " count=" << count);
                }
            }

            rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
            hipStream_t stream = static_cast<hipStream_t>(streams[i]);
            r = rccl::ncclAllReduce(
                buffers[i], buffers[i], count,
                toRcclDataTypeInt(toDataTypeInt(dtype)), toRcclRedOpInt(toOpInt(op)),
                comm, stream);
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rcclAllReduce(on-stream group) failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              rccl::ncclGetErrorString(r);
                rccl::ncclGroupEnd();
                return false;
            }
        }

        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        collective_performed_.store(true);
        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)op;
        (void)streams;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allreduceWithSidebandsMultiOnStreams(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op,
        const std::vector<CollectiveSidebandMultiOnStreamsOp> &sidebands,
        const std::vector<void *> &streams,
        const std::vector<NativeCollectiveRows> &live_rows)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (buffers.size() != static_cast<size_t>(num_devices_) ||
            streams.size() != static_cast<size_t>(num_devices_) ||
            (!live_rows.empty() && live_rows.size() != static_cast<size_t>(num_devices_)))
        {
            last_error_ = "Buffer/stream count does not match device count";
            return false;
        }

        for (int i = 0; i < num_devices_; ++i)
        {
            // Validate the complete bundle before opening a native group. Each
            // GPU owns a distinct count pointer, never a host-read row value.
            if (!live_rows.empty() &&
                (live_rows[i].bankElements() != count ||
                 live_rows[i].elementsPerRow() != live_rows.front().elementsPerRow() ||
                 (live_rows[i].rows().countOwner() != nullptr) !=
                     (live_rows.front().rows().countOwner() != nullptr) ||
                 !nativeCollectiveRowsValid(NativeRowCollective::AllReduce,
                     buffers[i], buffers[i], live_rows[i], dtype, op, num_devices_, i, streams[i])))
            {
                last_error_ = "Grouped allreduce live-row binding mismatch at participant " + std::to_string(i);
                return false;
            }
            if (!buffers[i])
            {
                last_error_ = "Null anchor allreduce buffer for device " + std::to_string(i);
                return false;
            }
            if (!streams[i])
            {
                last_error_ = "Null grouped bundle stream for device " + std::to_string(i);
                return false;
            }
        }

        for (size_t sideband_idx = 0; sideband_idx < sidebands.size(); ++sideband_idx)
        {
            const auto &sideband = sidebands[sideband_idx];
            if (sideband.count == 0)
            {
                last_error_ = "Zero-count grouped sideband " + std::to_string(sideband_idx);
                return false;
            }
            if (sideband.kind == CollectiveSidebandOp::Broadcast &&
                (sideband.root < 0 || sideband.root >= num_devices_))
            {
                last_error_ = "Invalid grouped sideband broadcast root " +
                              std::to_string(sideband.root);
                return false;
            }
            if (sideband.recv_buffers.size() != static_cast<size_t>(num_devices_))
            {
                last_error_ = "Grouped sideband recv buffer count mismatch at index " +
                              std::to_string(sideband_idx);
                return false;
            }
            if ((sideband.kind == CollectiveSidebandOp::Allgather ||
                 sideband.kind == CollectiveSidebandOp::Broadcast) &&
                sideband.send_buffers.size() != static_cast<size_t>(num_devices_))
            {
                last_error_ = "Grouped sideband send buffer count mismatch at index " +
                              std::to_string(sideband_idx);
                return false;
            }
            for (int i = 0; i < num_devices_; ++i)
            {
                if (!sideband.recv_buffers[static_cast<size_t>(i)])
                {
                    last_error_ = "Null grouped sideband recv buffer at sideband " +
                                  std::to_string(sideband_idx) + " device " + std::to_string(i);
                    return false;
                }
                if ((sideband.kind == CollectiveSidebandOp::Allgather ||
                     sideband.kind == CollectiveSidebandOp::Broadcast) &&
                    !sideband.send_buffers[static_cast<size_t>(i)])
                {
                    last_error_ = "Null grouped sideband send buffer at sideband " +
                                  std::to_string(sideband_idx) + " device " + std::to_string(i);
                    return false;
                }
            }
        }

        const bool trace_device_state = debugEnv().validation.validate_gpu_ptrs;
        const size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
        std::lock_guard<std::mutex> lock(direct_exec_mutex_);

        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                rccl::ncclGroupEnd();
                return false;
            }

            if (trace_device_state)
            {
                int current_device = -1;
                hipError_t get_device = hipGetDevice(&current_device);
                if (get_device == hipSuccess)
                {
                    LOG_DEBUG("[RCCL_STREAM_GROUP_BUNDLE] thread=" << thread_hash
                                                                    << " slot=" << i
                                                                    << " target_device=" << device_ordinals_[i]
                                                                    << " current_device=" << current_device
                                                                    << " stream=" << streams[i]
                                                                    << " buffer=" << buffers[i]
                                                                    << " count=" << count
                                                                    << " sidebands=" << sidebands.size());
                }
            }

            rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
            hipStream_t stream = static_cast<hipStream_t>(streams[i]);
            r = live_rows.empty()
                ? rccl::ncclAllReduce(buffers[i], buffers[i], count,
                    toRcclDataTypeInt(toDataTypeInt(dtype)), toRcclRedOpInt(toOpInt(op)), comm, stream)
                : rccl::nativeRows(NativeRowCollective::AllReduce,
                    buffers[i], buffers[i], live_rows[i],
                    toRcclDataTypeInt(toDataTypeInt(dtype)), toRcclRedOpInt(toOpInt(op)),
                    nullptr, comm, stream);
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rcclAllReduce(grouped bundle anchor) failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              rccl::ncclGetErrorString(r);
                rccl::ncclGroupEnd();
                return false;
            }
        }

        for (size_t sideband_idx = 0; sideband_idx < sidebands.size(); ++sideband_idx)
        {
            const auto &sideband = sidebands[sideband_idx];
            for (int i = 0; i < num_devices_; ++i)
            {
                hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                    rccl::ncclGroupEnd();
                    return false;
                }

                rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
                hipStream_t stream = static_cast<hipStream_t>(streams[i]);
                const auto rccl_dtype = toRcclDataTypeInt(toDataTypeInt(sideband.dtype));
                switch (sideband.kind)
                {
                case CollectiveSidebandOp::AllreduceSum:
                    r = rccl::ncclAllReduce(
                        sideband.recv_buffers[static_cast<size_t>(i)],
                        sideband.recv_buffers[static_cast<size_t>(i)],
                        sideband.count,
                        rccl_dtype,
                        rccl::ncclSum,
                        comm,
                        stream);
                    break;
                case CollectiveSidebandOp::Allgather:
                    r = rccl::ncclAllGather(
                        sideband.send_buffers[static_cast<size_t>(i)],
                        sideband.recv_buffers[static_cast<size_t>(i)],
                        sideband.count,
                        rccl_dtype,
                        comm,
                        stream);
                    break;
                case CollectiveSidebandOp::Broadcast:
                    r = rccl::ncclBroadcast(
                        sideband.send_buffers[static_cast<size_t>(i)],
                        sideband.recv_buffers[static_cast<size_t>(i)],
                        sideband.count,
                        rccl_dtype,
                        sideband.root,
                        comm,
                        stream);
                    break;
                }

                if (r != rccl::ncclSuccess)
                {
                    last_error_ = std::string("RCCL grouped sideband failed at sideband ") +
                                  std::to_string(sideband_idx) + " device " +
                                  std::to_string(device_ordinals_[i]) + ": " +
                                  rccl::ncclGetErrorString(r);
                    rccl::ncclGroupEnd();
                    return false;
                }
            }
        }

        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        collective_performed_.store(true);
        return true;
#else
        (void)buffers;
        (void)count;
        (void)dtype;
        (void)op;
        (void)sidebands;
        (void)streams;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::collectiveSidebandsMultiOnStreams(
        const std::vector<CollectiveSidebandMultiOnStreamsOp> &sidebands,
        const std::vector<void *> &streams)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }
        if (sidebands.empty())
        {
            last_error_ =
                "collectiveSidebandsMultiOnStreams requires at least one sideband";
            return false;
        }
        if (streams.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Stream count does not match device count";
            return false;
        }
        for (int i = 0; i < num_devices_; ++i)
        {
            if (!streams[static_cast<size_t>(i)])
            {
                last_error_ = "Null grouped sideband stream for device " +
                              std::to_string(i);
                return false;
            }
        }

        /*
         * Reject malformed publication contracts before opening the RCCL group.
         * A participant mismatch discovered after the first launch would leave a
         * partial collective sequence resident in the communicator.
         */
        for (size_t sideband_index = 0;
             sideband_index < sidebands.size();
             ++sideband_index)
        {
            const auto &sideband = sidebands[sideband_index];
            if (sideband.count == 0 ||
                sideband.recv_buffers.size() !=
                    static_cast<size_t>(num_devices_))
            {
                last_error_ =
                    "Invalid grouped sideband descriptor at index " +
                    std::to_string(sideband_index);
                return false;
            }
            if (sideband.kind == CollectiveSidebandOp::Broadcast &&
                (sideband.root < 0 || sideband.root >= num_devices_))
            {
                last_error_ = "Invalid grouped sideband broadcast root " +
                              std::to_string(sideband.root);
                return false;
            }
            const bool requires_send_buffers =
                sideband.kind == CollectiveSidebandOp::Allgather ||
                sideband.kind == CollectiveSidebandOp::Broadcast;
            if (requires_send_buffers &&
                sideband.send_buffers.size() !=
                    static_cast<size_t>(num_devices_))
            {
                last_error_ =
                    "Grouped sideband send buffer count mismatch at index " +
                    std::to_string(sideband_index);
                return false;
            }
            for (int i = 0; i < num_devices_; ++i)
            {
                if (!sideband.recv_buffers[static_cast<size_t>(i)] ||
                    (requires_send_buffers &&
                     !sideband.send_buffers[static_cast<size_t>(i)]))
                {
                    last_error_ =
                        "Null grouped sideband buffer at sideband " +
                        std::to_string(sideband_index) + " device " +
                        std::to_string(i);
                    return false;
                }
            }
        }

        std::lock_guard<std::mutex> lock(direct_exec_mutex_);
        rccl::ncclResult_t result = rccl::ncclGroupStart();
        if (result != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") +
                          rccl::ncclGetErrorString(result);
            return false;
        }

        for (size_t sideband_index = 0;
             sideband_index < sidebands.size();
             ++sideband_index)
        {
            const auto &sideband = sidebands[sideband_index];
            for (int i = 0; i < num_devices_; ++i)
            {
                const hipError_t set_device =
                    trackedHipSetDevice(
                        device_ordinals_[static_cast<size_t>(i)]);
                if (set_device != hipSuccess)
                {
                    last_error_ = std::string("hipSetDevice failed: ") +
                                  hipGetErrorString(set_device);
                    rccl::ncclGroupEnd();
                    return false;
                }

                const auto comm =
                    static_cast<rccl::ncclComm_t>(
                        comms_[static_cast<size_t>(i)]);
                const auto stream =
                    static_cast<hipStream_t>(
                        streams[static_cast<size_t>(i)]);
                const auto dtype =
                    toRcclDataTypeInt(toDataTypeInt(sideband.dtype));
                switch (sideband.kind)
                {
                case CollectiveSidebandOp::AllreduceSum:
                    result = rccl::ncclAllReduce(
                        sideband.recv_buffers[static_cast<size_t>(i)],
                        sideband.recv_buffers[static_cast<size_t>(i)],
                        sideband.count,
                        dtype,
                        rccl::ncclSum,
                        comm,
                        stream);
                    break;
                case CollectiveSidebandOp::Allgather:
                    result = rccl::ncclAllGather(
                        sideband.send_buffers[static_cast<size_t>(i)],
                        sideband.recv_buffers[static_cast<size_t>(i)],
                        sideband.count,
                        dtype,
                        comm,
                        stream);
                    break;
                case CollectiveSidebandOp::Broadcast:
                    result = rccl::ncclBroadcast(
                        sideband.send_buffers[static_cast<size_t>(i)],
                        sideband.recv_buffers[static_cast<size_t>(i)],
                        sideband.count,
                        dtype,
                        sideband.root,
                        comm,
                        stream);
                    break;
                }
                if (result != rccl::ncclSuccess)
                {
                    last_error_ =
                        "RCCL grouped sideband publication failed at sideband " +
                        std::to_string(sideband_index) + " device " +
                        std::to_string(device_ordinals_[static_cast<size_t>(i)]) +
                        ": " + rccl::ncclGetErrorString(result);
                    rccl::ncclGroupEnd();
                    return false;
                }
            }
        }

        result = rccl::ncclGroupEnd();
        if (result != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") +
                          rccl::ncclGetErrorString(result);
            return false;
        }
        collective_performed_.store(true);
        return true;
#else
        (void)sidebands;
        (void)streams;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allreduceSingleDeviceAsync(void *buffer, size_t count,
                                                     CollectiveDataType dtype, CollectiveOp op,
                                                     int device_idx)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
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

        // Require compute streams for async path
        if (compute_streams_.empty() ||
            static_cast<int>(compute_streams_.size()) != num_devices_)
        {
            last_error_ = "Compute streams not registered — call setComputeStreams() first";
            return false;
        }

        // Per-device resources (no locking needed — each device_idx is accessed
        // by exactly one thread in the barrier-free TP path)
        const int ordinal = device_ordinals_[device_idx];
        hipStream_t compute_stream = static_cast<hipStream_t>(compute_streams_[device_idx]);
        hipEvent_t compute_event = static_cast<hipEvent_t>(compute_events_[device_idx]);
        hipStream_t rccl_stream = static_cast<hipStream_t>(streams_[device_idx]);
        hipEvent_t completion_event = static_cast<hipEvent_t>(completion_events_[device_idx]);
        rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[device_idx]);

        hipError_t err;

        // 1. Set device context
        err = trackedHipSetDevice(ordinal);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
            return false;
        }

        // 2. Pre-sync: record event on compute stream, RCCL stream waits for it
        err = hipEventRecord(compute_event, compute_stream);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipEventRecord(compute) failed: ") + hipGetErrorString(err);
            return false;
        }

        err = hipStreamWaitEvent(rccl_stream, compute_event, 0);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipStreamWaitEvent(rccl←compute) failed: ") + hipGetErrorString(err);
            return false;
        }

        // 3. Launch allreduce (non-grouped — RCCL matches calls internally)
        rccl::ncclResult_t r = rccl::ncclAllReduce(
            buffer, buffer, count,
            toRcclDataTypeInt(toDataTypeInt(dtype)), toRcclRedOpInt(toOpInt(op)),
            comm, rccl_stream);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclAllReduce failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }
        collective_performed_.store(true);

        // 4. Post-sync: record completion on RCCL stream, compute stream waits
        err = hipEventRecord(completion_event, rccl_stream);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipEventRecord(completion) failed: ") + hipGetErrorString(err);
            return false;
        }

        err = hipStreamWaitEvent(compute_stream, completion_event, 0);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipStreamWaitEvent(compute←rccl) failed: ") + hipGetErrorString(err);
            return false;
        }

        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allreduceSingleDeviceOnStream(void *buffer, size_t count,
                                                        CollectiveDataType dtype, CollectiveOp op,
                                                        int device_idx, void *stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
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
        rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[device_idx]);
        hipStream_t caller_stream = static_cast<hipStream_t>(stream);

        // Fast path: skip hipSetDevice if this thread already has the right device.
        // In LOCAL TP decode, each worker thread always targets the same device,
        // so after the first call the branch is always taken (~1-3μs saved per call).
        static thread_local int tl_last_hip_device = -1;
        if (tl_last_hip_device != ordinal)
        {
            hipError_t err = trackedHipSetDevice(ordinal);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                return false;
            }
            tl_last_hip_device = ordinal;
        }

        // 2. Launch allreduce directly on the caller's stream.
        //    No cross-stream event sync needed — the caller's stream provides
        //    ordering (prior compute → allreduce → subsequent compute).
        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=rccl_onstream_launch"
                     << " coordinator=" << static_cast<const void *>(this)
                     << " slot=" << device_idx
                     << " ordinal=" << ordinal
                     << " buffer=" << buffer
                     << " count=" << count
                     << " dtype=" << static_cast<int>(dtype)
                     << " op=" << static_cast<int>(op)
                     << " stream=" << stream
                     << " comm=" << comm);
        }
        rccl::ncclResult_t r = rccl::ncclAllReduce(
            buffer, buffer, count,
            toRcclDataTypeInt(toDataTypeInt(dtype)), toRcclRedOpInt(toOpInt(op)),
            comm, caller_stream);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclAllReduce(on-stream) failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }
        collective_performed_.store(true);

        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=rccl_onstream_enqueued"
                     << " coordinator=" << static_cast<const void *>(this)
                     << " slot=" << device_idx
                     << " ordinal=" << ordinal
                     << " stream=" << stream);
        }

        return true;
#else
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)op;
        (void)device_idx;
        (void)stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::reduceSingleDeviceOnStream(
        const void *send_buf,
        void *recv_buf,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op,
        int root,
        int device_idx,
        void *stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }
        if (device_idx < 0 || device_idx >= num_devices_)
        {
            last_error_ = "Invalid device_idx " + std::to_string(device_idx) +
                          " (num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }
        if (root < 0 || root >= num_devices_)
        {
            last_error_ = "Invalid reduce root " + std::to_string(root) +
                          " (num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }
        if (!send_buf || !recv_buf || count == 0)
        {
            last_error_ = "Invalid reduce buffer/count for device " +
                          std::to_string(device_idx);
            return false;
        }
        if (!stream)
        {
            last_error_ = "Null reduce stream for device " +
                          std::to_string(device_idx);
            return false;
        }

        const int ordinal = device_ordinals_[device_idx];
        const auto comm =
            static_cast<rccl::ncclComm_t>(comms_[device_idx]);
        const auto caller_stream = static_cast<hipStream_t>(stream);

        static thread_local int tl_last_hip_device_for_reduce = -1;
        if (tl_last_hip_device_for_reduce != ordinal)
        {
            const hipError_t err = trackedHipSetDevice(ordinal);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") +
                              hipGetErrorString(err);
                return false;
            }
            tl_last_hip_device_for_reduce = ordinal;
        }

        const rccl::ncclResult_t result = rccl::ncclReduce(
            send_buf,
            recv_buf,
            count,
            toRcclDataTypeInt(toDataTypeInt(dtype)),
            toRcclRedOpInt(toOpInt(op)),
            root,
            comm,
            caller_stream);
        if (result != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclReduce(on-stream) failed: ") +
                          rccl::ncclGetErrorString(result);
            return false;
        }
        collective_performed_.store(true);
        return true;
#else
        (void)send_buf;
        (void)recv_buf;
        (void)count;
        (void)dtype;
        (void)op;
        (void)root;
        (void)device_idx;
        (void)stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::nativeRowsOnStream(
        NativeRowCollective operation, const void *send, void *receive,
        const NativeCollectiveRows &rows, CollectiveDataType dtype,
        CollectiveOp reduction, int participant, void *stream, unsigned long long *payload_bytes)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load() || !nativeCollectiveRowsValid(operation, send, receive, rows,
                dtype, reduction, num_devices_, participant, stream))
        {
            last_error_ = "RCCL live-row collective has invalid native row/buffer/stream geometry";
            return false;
        }
        // Another operation may change this host thread's current GPU between
        // graph recordings. Select the actual owner; never cache that assumption.
        const auto selected = trackedHipSetDevice(device_ordinals_[participant]);
        if (selected != hipSuccess)
        {
            last_error_ = std::string("RCCL live-row device selection failed: ") + hipGetErrorString(selected);
            return false;
        }
        const auto producer_error = hipGetLastError();
        if (producer_error != hipSuccess)
        {
            last_error_ = std::string("RCCL live-row producer failed: ") + hipGetErrorString(producer_error);
            return false;
        }
        const auto native_op = operation == NativeRowCollective::AllGather ? rccl::ncclSum :
            toRcclRedOpInt(toOpInt(reduction));
        const auto result = rccl::nativeRows(operation, send, receive, rows,
            toRcclDataTypeInt(toDataTypeInt(dtype)), native_op, payload_bytes,
            static_cast<rccl::ncclComm_t>(comms_[participant]), stream);
        if (result != rccl::ncclSuccess)
        {
            last_error_ = std::string("RCCL live-row enqueue failed: ") + rccl::ncclGetErrorString(result);
            return false;
        }
        const auto enqueue_error = hipGetLastError();
        if (enqueue_error != hipSuccess)
        {
            last_error_ = std::string("RCCL live-row runtime enqueue failed: ") + hipGetErrorString(enqueue_error);
            return false;
        }
        collective_performed_.store(true);
        return true;
#else
        (void)operation; (void)send; (void)receive; (void)rows; (void)dtype;
        (void)reduction; (void)participant; (void)stream; (void)payload_bytes;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::reduceScatterSingleDeviceOnStream(
        const void *send_buf, void *recv_buf, size_t receive_count,
        CollectiveDataType dtype, int device_idx, void *stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load() || !nativeReduceScatterBuffersValid(
                send_buf, recv_buf, receive_count, dtype, num_devices_, device_idx, stream))
        {
            last_error_ = "RCCL reduce-scatter has an invalid communicator, stream or disjoint buffer geometry";
            return false;
        }
        // Never cache an assumed current device here: another operation on
        // this host thread may have selected a different participant since the
        // last capture. Replay has no host-side device-selection work.
        const auto selected = trackedHipSetDevice(device_ordinals_[device_idx]);
        if (selected != hipSuccess)
        {
            last_error_ = std::string("hipSetDevice before reduce-scatter failed: ") + hipGetErrorString(selected);
            return false;
        }
        const auto producer_error = hipGetLastError();
        if (producer_error != hipSuccess)
        {
            last_error_ = std::string("RCCL reduce-scatter producer launch failed: ") + hipGetErrorString(producer_error);
            return false;
        }
        const auto result = rccl::ncclReduceScatter(send_buf, recv_buf, receive_count,
            toRcclDataTypeInt(toDataTypeInt(dtype)), rccl::ncclSum,
            static_cast<rccl::ncclComm_t>(comms_[device_idx]), static_cast<hipStream_t>(stream));
        if (result != rccl::ncclSuccess)
        {
            last_error_ = std::string("RCCL native reduce-scatter failed: ") + rccl::ncclGetErrorString(result);
            return false;
        }
        // Attribute sticky enqueue failures here, not to the next model kernel.
        // Later asynchronous execution errors remain owned by graph completion.
        const auto enqueue_error = hipGetLastError();
        if (enqueue_error != hipSuccess)
        {
            last_error_ = std::string("RCCL reduce-scatter runtime enqueue failed: ") + hipGetErrorString(enqueue_error);
            return false;
        }
        collective_performed_.store(true);
        return true;
#else
        (void)send_buf; (void)recv_buf; (void)receive_count;
        (void)dtype; (void)device_idx; (void)stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allgatherSingleDeviceOnStream(const void *send_buf,
                                                        void *recv_buf,
                                                        size_t send_count,
                                                        CollectiveDataType dtype,
                                                        int device_idx,
                                                        void *stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (device_idx < 0 || device_idx >= num_devices_)
        {
            last_error_ = "Invalid device_idx " + std::to_string(device_idx) +
                          " (num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        if (!send_buf || !recv_buf)
        {
            last_error_ = "Null allgather buffer for device " + std::to_string(device_idx);
            return false;
        }

        if (!stream)
        {
            last_error_ = "Null stream for device " + std::to_string(device_idx);
            return false;
        }

        const int ordinal = device_ordinals_[device_idx];
        rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[device_idx]);
        hipStream_t caller_stream = static_cast<hipStream_t>(stream);

        static thread_local int tl_last_hip_device_for_allgather = -1;
        if (tl_last_hip_device_for_allgather != ordinal)
        {
            hipError_t err = trackedHipSetDevice(ordinal);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                return false;
            }
            tl_last_hip_device_for_allgather = ordinal;
        }

        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=rccl_allgather_onstream_launch"
                     << " coordinator=" << static_cast<const void *>(this)
                     << " slot=" << device_idx
                     << " ordinal=" << ordinal
                     << " send_buf=" << send_buf
                     << " recv_buf=" << recv_buf
                     << " count=" << send_count
                     << " dtype=" << static_cast<int>(dtype)
                     << " stream=" << stream
                     << " comm=" << comm);
        }

        /*
         * HIP launch errors are thread-local and sticky.  Attribute producer
         * and RCCL enqueue failures at this primitive instead of allowing the
         * next unrelated kernel wrapper to consume and mislabel them.  This
         * check is fail-fast only: no retry, synchronization, or alternate
         * transport is permitted.
         */
        const hipError_t producer_error = hipGetLastError();
        if (producer_error != hipSuccess)
        {
            last_error_ =
                std::string("HIP producer launch state failed before rcclAllGather(on-stream): ") +
                hipGetErrorString(producer_error);
            return false;
        }

        rccl::ncclResult_t r = rccl::ncclAllGather(
            send_buf,
            recv_buf,
            send_count,
            toRcclDataTypeInt(toDataTypeInt(dtype)),
            comm,
            caller_stream);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclAllGather(on-stream) failed: ") +
                          rccl::ncclGetErrorString(r);
            return false;
        }
        const hipError_t collective_launch_error = hipGetLastError();
        if (collective_launch_error != hipSuccess)
        {
            last_error_ =
                std::string("HIP runtime rejected rcclAllGather(on-stream) enqueue: ") +
                hipGetErrorString(collective_launch_error);
            return false;
        }

        collective_performed_.store(true);
        return true;
#else
        (void)send_buf;
        (void)recv_buf;
        (void)send_count;
        (void)dtype;
        (void)device_idx;
        (void)stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::broadcastSingleDeviceOnStream(const void *send_buf,
                                                        void *recv_buf,
                                                        size_t count,
                                                        CollectiveDataType dtype,
                                                        int root,
                                                        int device_idx,
                                                        void *stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (device_idx < 0 || device_idx >= num_devices_)
        {
            last_error_ = "Invalid device_idx " + std::to_string(device_idx) +
                          " (num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        if (root < 0 || root >= num_devices_)
        {
            last_error_ = "Invalid broadcast root " + std::to_string(root) +
                          " (num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        if (!send_buf || !recv_buf)
        {
            last_error_ = "Null broadcast buffer for device " + std::to_string(device_idx);
            return false;
        }

        if (!stream)
        {
            last_error_ = "Null stream for device " + std::to_string(device_idx);
            return false;
        }

        const int ordinal = device_ordinals_[device_idx];
        rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[device_idx]);
        hipStream_t caller_stream = static_cast<hipStream_t>(stream);

        static thread_local int tl_last_hip_device_for_broadcast = -1;
        if (tl_last_hip_device_for_broadcast != ordinal)
        {
            hipError_t err = trackedHipSetDevice(ordinal);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                return false;
            }
            tl_last_hip_device_for_broadcast = ordinal;
        }

        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=rccl_broadcast_onstream_launch"
                     << " coordinator=" << static_cast<const void *>(this)
                     << " slot=" << device_idx
                     << " ordinal=" << ordinal
                     << " send_buf=" << send_buf
                     << " recv_buf=" << recv_buf
                     << " count=" << count
                     << " dtype=" << static_cast<int>(dtype)
                     << " root=" << root
                     << " stream=" << stream
                     << " comm=" << comm);
        }

        rccl::ncclResult_t r = rccl::ncclBroadcast(
            send_buf,
            recv_buf,
            count,
            toRcclDataTypeInt(toDataTypeInt(dtype)),
            root,
            comm,
            caller_stream);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclBroadcast(on-stream) failed: ") +
                          rccl::ncclGetErrorString(r);
            return false;
        }

        collective_performed_.store(true);
        return true;
#else
        (void)send_buf;
        (void)recv_buf;
        (void)count;
        (void)dtype;
        (void)root;
        (void)device_idx;
        (void)stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::groupedP2PSingleDeviceOnStream(
        const std::vector<CollectiveP2POp> &ops,
        int device_idx,
        void *stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }
        if (device_idx < 0 || device_idx >= num_devices_)
        {
            last_error_ = "Invalid device_idx " + std::to_string(device_idx) +
                          " (num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }
        if (!stream)
        {
            last_error_ = "Null stream for grouped P2P device " + std::to_string(device_idx);
            return false;
        }
        if (ops.empty())
            return true;

        const int ordinal = device_ordinals_[device_idx];
        rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[device_idx]);
        hipStream_t caller_stream = static_cast<hipStream_t>(stream);

        static thread_local int tl_last_hip_device_for_grouped_p2p = -1;
        if (tl_last_hip_device_for_grouped_p2p != ordinal)
        {
            hipError_t err = trackedHipSetDevice(ordinal);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                return false;
            }
            tl_last_hip_device_for_grouped_p2p = ordinal;
        }

        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=rccl_grouped_p2p_onstream_launch"
                     << " coordinator=" << static_cast<const void *>(this)
                     << " slot=" << device_idx
                     << " ordinal=" << ordinal
                     << " ops=" << ops.size()
                     << " stream=" << stream
                     << " comm=" << comm);
        }

        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart(grouped P2P) failed: ") +
                          rccl::ncclGetErrorString(r);
            return false;
        }

        for (const auto &op : ops)
        {
            if (op.peer < 0 || op.peer >= num_devices_ || op.peer == device_idx)
            {
                last_error_ = "Invalid grouped P2P peer " + std::to_string(op.peer) +
                              " for device " + std::to_string(device_idx);
                rccl::ncclGroupEnd();
                return false;
            }
            const auto dtype_int = toRcclDataTypeInt(toDataTypeInt(op.dtype));
            if (op.kind == CollectiveP2POpKind::Send)
            {
                if (!op.send_buffer || op.count == 0)
                {
                    last_error_ = "Invalid grouped P2P send buffer/count";
                    rccl::ncclGroupEnd();
                    return false;
                }
                r = rccl::ncclSend(
                    op.send_buffer,
                    op.count,
                    dtype_int,
                    op.peer,
                    comm,
                    caller_stream);
            }
            else
            {
                if (!op.recv_buffer || op.count == 0)
                {
                    last_error_ = "Invalid grouped P2P recv buffer/count";
                    rccl::ncclGroupEnd();
                    return false;
                }
                r = rccl::ncclRecv(
                    op.recv_buffer,
                    op.count,
                    dtype_int,
                    op.peer,
                    comm,
                    caller_stream);
            }
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rccl grouped P2P op failed: ") +
                              rccl::ncclGetErrorString(r);
                rccl::ncclGroupEnd();
                return false;
            }
        }

        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd(grouped P2P) failed: ") +
                          rccl::ncclGetErrorString(r);
            return false;
        }
        collective_performed_.store(true);
        return true;
#else
        (void)ops;
        (void)device_idx;
        (void)stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::broadcastMultiOnStreams(const std::vector<const void *> &send_buffers,
                                                  const std::vector<void *> &recv_buffers,
                                                  size_t count,
                                                  CollectiveDataType dtype,
                                                  int root,
                                                  const std::vector<void *> &streams)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (root < 0 || root >= num_devices_)
        {
            last_error_ = "Invalid broadcast root " + std::to_string(root);
            return false;
        }

        if (send_buffers.size() != static_cast<size_t>(num_devices_) ||
            recv_buffers.size() != static_cast<size_t>(num_devices_) ||
            streams.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer/stream count does not match device count";
            return false;
        }

        for (int i = 0; i < num_devices_; ++i)
        {
            if (!send_buffers[i] || !recv_buffers[i])
            {
                last_error_ = "Null broadcast buffer for device " + std::to_string(i);
                return false;
            }
            if (!streams[i])
            {
                last_error_ = "Null broadcast stream for device " + std::to_string(i);
                return false;
            }
        }

        const bool trace_device_state = debugEnv().validation.validate_gpu_ptrs;
        const size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
        std::lock_guard<std::mutex> lock(direct_exec_mutex_);

        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                rccl::ncclGroupEnd();
                return false;
            }

            if (trace_device_state)
            {
                int current_device = -1;
                hipError_t get_device = hipGetDevice(&current_device);
                if (get_device == hipSuccess)
                {
                    LOG_DEBUG("[RCCL_STREAM_GROUP_BROADCAST] thread=" << thread_hash
                                                                       << " slot=" << i
                                                                       << " target_device=" << device_ordinals_[i]
                                                                       << " current_device=" << current_device
                                                                       << " stream=" << streams[i]
                                                                       << " send=" << send_buffers[i]
                                                                       << " recv=" << recv_buffers[i]
                                                                       << " count=" << count
                                                                       << " root=" << root);
                }
            }

            rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
            hipStream_t stream = static_cast<hipStream_t>(streams[i]);
            r = rccl::ncclBroadcast(
                send_buffers[i],
                recv_buffers[i],
                count,
                toRcclDataTypeInt(toDataTypeInt(dtype)),
                root,
                comm,
                stream);
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rcclBroadcast(on-stream group) failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " +
                              rccl::ncclGetErrorString(r);
                rccl::ncclGroupEnd();
                return false;
            }
        }

        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        collective_performed_.store(true);
        return true;
#else
        (void)send_buffers;
        (void)recv_buffers;
        (void)count;
        (void)dtype;
        (void)root;
        (void)streams;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allgatherMulti(const std::vector<const void *> &send_buffers,
                                         const std::vector<void *> &recv_buffers,
                                         size_t send_count, CollectiveDataType dtype)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
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
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allgatherMultiWithComputeDeps(
        const std::vector<const void *> &send_buffers,
        const std::vector<void *> &recv_buffers,
        size_t send_count,
        CollectiveDataType dtype)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (send_buffers.size() != static_cast<size_t>(num_devices_) ||
            recv_buffers.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = "Buffer count doesn't match device count";
            return false;
        }

        if (compute_streams_.empty() ||
            static_cast<int>(compute_streams_.size()) != num_devices_)
        {
            LOG_DEBUG("[RCCLCoordinator] allgatherMultiWithComputeDeps: no compute streams, "
                      "falling back to synchronous allgather");
            return submitAndWait([&]()
                                 {
                if (!doAllgatherMulti(send_buffers, recv_buffers, send_count, toDataTypeInt(dtype)))
                    return false;
                return doSynchronizeAll(); });
        }

        {
            std::lock_guard<std::mutex> lock(direct_exec_mutex_);
            if (!doAllgatherMulti(send_buffers, recv_buffers, send_count, toDataTypeInt(dtype)))
                return false;
            return doInsertComputeStreamDeps();
        }
#else
        (void)send_buffers;
        (void)recv_buffers;
        (void)send_count;
        (void)dtype;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::broadcastMulti(const std::vector<void *> &buffers, size_t count,
                                         CollectiveDataType dtype, int root)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
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
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::reduceScatterMulti(const std::vector<const void *> &send_buffers,
                                             const std::vector<void *> &recv_buffers,
                                             size_t recv_count, CollectiveDataType dtype,
                                             CollectiveOp op)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
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
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::allreduce(void *buffer, size_t count, CollectiveDataType dtype,
                                    CollectiveOp op, int device_idx)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (device_idx < 0 || device_idx >= num_devices_)
        {
            last_error_ = "Invalid device_idx: " + std::to_string(device_idx);
            return false;
        }

        return submitAndWait([&]()
                             {
        hipError_t err = trackedHipSetDevice(device_ordinals_[device_idx]);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
            return false;
        }

        rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[device_idx]);
        hipStream_t stream = static_cast<hipStream_t>(streams_[device_idx]);

        rccl::ncclResult_t r = rccl::ncclAllReduce(buffer, buffer, count,
                                                   toRcclDataType(dtype), toRcclRedOp(op),
                                                   comm, stream);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclAllReduce failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }
        collective_performed_.store(true);

        // Record completion event
        err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[device_idx]), stream);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipEventRecord failed: ") + hipGetErrorString(err);
            return false;
        }

        return true; });
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::synchronize()
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        return submitAndWait([&]()
                             { return doSynchronizeAll(); });
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::doSynchronizeAll()
    {
#ifdef HAVE_RCCL
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                return false;
            }

            // Stream-sync on RCCL stream only (not all streams).
            // RCCL guarantees that when work completes on the user-provided stream,
            // all internal RCCL work is also complete. hipDeviceSynchronize() was
            // overkill — it stalls ALL streams including the compute stream.
            hipStream_t rccl_stream = static_cast<hipStream_t>(streams_[i]);
            err = hipStreamSynchronize(rccl_stream);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipStreamSynchronize failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                return false;
            }
        }
        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::doInsertComputeStreamDeps()
    {
#ifdef HAVE_RCCL
        // Insert GPU-side dependencies: make each compute stream wait for the
        // RCCL completion event. This ensures the compute stream cannot execute
        // post-allreduce kernels until RCCL has finished, WITHOUT blocking the
        // host thread. The host returns immediately after these API calls.
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed in doInsertComputeStreamDeps: ") +
                              hipGetErrorString(err);
                return false;
            }

            hipStream_t compute_stream = static_cast<hipStream_t>(compute_streams_[i]);
            hipEvent_t completion_event = static_cast<hipEvent_t>(completion_events_[i]);

            err = hipStreamWaitEvent(compute_stream, completion_event, 0);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipStreamWaitEvent(compute, rccl_completion) failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[RCCLCoordinator] Inserted compute stream deps for " << num_devices_ << " devices");
        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::copy(void *dst_ptr, int dst_device_idx,
                               const void *src_ptr, int src_device_idx,
                               size_t bytes)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load())
        {
            last_error_ = "RCCLCoordinator not initialized";
            return false;
        }

        if (bytes == 0)
        {
            return true; // No-op for zero bytes
        }

        if (!dst_ptr || !src_ptr)
        {
            last_error_ = "RCCLCoordinator::copy: null buffer pointer";
            return false;
        }

        if (dst_device_idx < 0 || dst_device_idx >= num_devices_ ||
            src_device_idx < 0 || src_device_idx >= num_devices_)
        {
            last_error_ = "RCCLCoordinator::copy: device index out of range (src=" +
                          std::to_string(src_device_idx) + " dst=" + std::to_string(dst_device_idx) +
                          " num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        // Same device - use hipMemcpyAsync on coordinator stream (synchronous wait)
        if (src_device_idx == dst_device_idx)
        {
            return submitAndWait([&]()
                                 {
            hipError_t err = trackedHipSetDevice(device_ordinals_[src_device_idx]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                return false;
            }

            hipStream_t stream = static_cast<hipStream_t>(streams_[src_device_idx]);
            err = hipMemcpyAsync(dst_ptr, src_ptr, bytes, hipMemcpyDeviceToDevice, stream);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipMemcpyAsync failed: ") + hipGetErrorString(err);
                return false;
            }
            err = hipStreamSynchronize(stream);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipStreamSynchronize failed: ") + hipGetErrorString(err);
                return false;
            }
            return true; });
        }

        // Different devices - use RCCL send/recv with synchronization
        return submitAndWait([&]()
                             { return doCopy(dst_ptr, dst_device_idx, src_ptr, src_device_idx, bytes, /*wait_for_completion=*/true,
                                             streams_[src_device_idx], streams_[dst_device_idx]); });
#else
        (void)dst_ptr;
        (void)dst_device_idx;
        (void)src_ptr;
        (void)src_device_idx;
        (void)bytes;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::copyOnStreams(
        void *dst_ptr, int dst_device_idx,
        const void *src_ptr, int src_device_idx,
        size_t bytes, void *source_stream, void *destination_stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load() || !dst_ptr || !src_ptr || bytes == 0u ||
            !source_stream || !destination_stream ||
            dst_device_idx < 0 || dst_device_idx >= num_devices_ ||
            src_device_idx < 0 || src_device_idx >= num_devices_ ||
            dst_device_idx == src_device_idx)
        {
            last_error_ = "RCCLCoordinator::copyOnStreams: invalid cross-device stream binding";
            return false;
        }
        // A queue round trip serializes communicator use, not GPU completion.
        // Send and receive remain ordered solely by the supplied device DAG.
        return submitAndWait([&]() {
            return doCopy(dst_ptr, dst_device_idx, src_ptr, src_device_idx,
                          bytes, false, source_stream, destination_stream);
        });
#else
        (void)dst_ptr;
        (void)dst_device_idx;
        (void)src_ptr;
        (void)src_device_idx;
        (void)bytes;
        (void)source_stream;
        (void)destination_stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    // ============================================================================
    // Internal Collective Implementations (called ON coordinator thread)
    // ============================================================================

    bool RCCLCoordinator::doInsertCollectiveInputDeps(const char *operation)
    {
#ifdef HAVE_RCCL
        if (compute_streams_.size() != static_cast<size_t>(num_devices_) ||
            compute_events_.size() != static_cast<size_t>(num_devices_))
        {
            last_error_ = std::string(operation) +
                          " requires one registered compute stream and event per device";
            return false;
        }

        for (int i = 0; i < num_devices_; ++i)
        {
            if (!compute_streams_[i] || !compute_events_[i] || !streams_[i])
            {
                last_error_ = std::string(operation) +
                              " encountered an uninitialized stream/event slot for device " +
                              std::to_string(device_ordinals_[i]);
                return false;
            }

            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed before ") +
                              operation + ": " + hipGetErrorString(err);
                return false;
            }

            auto compute_stream =
                static_cast<hipStream_t>(compute_streams_[i]);
            auto compute_event =
                static_cast<hipEvent_t>(compute_events_[i]);
            auto collective_stream =
                static_cast<hipStream_t>(streams_[i]);
            err = hipEventRecord(compute_event, compute_stream);
            if (err == hipSuccess)
                err = hipStreamWaitEvent(collective_stream, compute_event, 0);
            if (err != hipSuccess)
            {
                last_error_ = std::string("HIP event handoff failed before ") +
                              operation + " on device " +
                              std::to_string(device_ordinals_[i]) + ": " +
                              hipGetErrorString(err);
                return false;
            }
        }
        return true;
#else
        (void)operation;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::doAllreduceMulti(const std::vector<void *> &buffers, size_t count,
                                           int dtype_int, int op_int)
    {
#ifdef HAVE_RCCL
        const bool trace_device_state = debugEnv().validation.validate_gpu_ptrs;
        const size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());

        if (!doInsertCollectiveInputDeps("RCCL allreduce"))
            return false;

        // Start RCCL group for multi-GPU operation
        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Issue allreduce for each device
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                rccl::ncclGroupEnd();
                return false;
            }

            if (trace_device_state)
            {
                int after_dev = -1;
                hipError_t get_after = hipGetDevice(&after_dev);
                if (get_after == hipSuccess && after_dev != device_ordinals_[i])
                {
                    LOG_ERROR("[RCCL_DEVICE_STATE_MISMATCH] phase=post_set_launch thread=" << thread_hash
                                                                                           << " slot=" << i
                                                                                           << " expected=" << device_ordinals_[i]
                                                                                           << " actual=" << after_dev);
                    last_error_ = "RCCLCoordinator device mismatch after hipSetDevice (launch)";
                    rccl::ncclGroupEnd();
                    return false;
                }
            }

            rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
            hipStream_t stream = static_cast<hipStream_t>(streams_[i]);

            if (trace_device_state)
            {
                int launch_dev = -1;
                hipError_t get_launch_dev = hipGetDevice(&launch_dev);
                if (get_launch_dev == hipSuccess)
                {
                    LOG_DEBUG("[RCCL_STREAM_LAUNCH] thread=" << thread_hash
                                                             << " slot=" << i
                                                             << " target_device=" << device_ordinals_[i]
                                                             << " current_device=" << launch_dev
                                                             << " stream=" << stream
                                                             << " buffer=" << buffers[i]
                                                             << " count=" << count);
                }
            }

            r = rccl::ncclAllReduce(buffers[i], buffers[i], count,
                                    toRcclDataTypeInt(dtype_int), toRcclRedOpInt(op_int),
                                    comm, stream);
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rcclAllReduce failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + rccl::ncclGetErrorString(r);
                rccl::ncclGroupEnd();
                return false;
            }
        }

        // End RCCL group
        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion events for all devices
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice for event record failed: ") + hipGetErrorString(err);
                return false;
            }

            err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[i]),
                                 static_cast<hipStream_t>(streams_[i]));
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipEventRecord failed: ") + hipGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[RCCLCoordinator] AllreduceMulti completed: " << count << " elements");
        collective_performed_.store(true);
        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::doAllgatherMulti(const std::vector<const void *> &send_buffers,
                                           const std::vector<void *> &recv_buffers,
                                           size_t send_count, int dtype_int)
    {
#ifdef HAVE_RCCL
        if (!doInsertCollectiveInputDeps("RCCL allgather"))
            return false;

        // Start RCCL group
        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Issue allgather for each device
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                rccl::ncclGroupEnd();
                return false;
            }

            rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
            hipStream_t stream = static_cast<hipStream_t>(streams_[i]);

            r = rccl::ncclAllGather(send_buffers[i], recv_buffers[i], send_count,
                                    toRcclDataTypeInt(dtype_int), comm, stream);
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rcclAllGather failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + rccl::ncclGetErrorString(r);
                rccl::ncclGroupEnd();
                return false;
            }
        }

        // End RCCL group
        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion events
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice for event record failed: ") + hipGetErrorString(err);
                return false;
            }

            err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[i]),
                                 static_cast<hipStream_t>(streams_[i]));
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipEventRecord failed: ") + hipGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[RCCLCoordinator] AllgatherMulti completed: " << send_count << " elements per device");
        collective_performed_.store(true);
        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::doBroadcastMulti(const std::vector<void *> &buffers, size_t count,
                                           int dtype_int, int root)
    {
#ifdef HAVE_RCCL
        if (!doInsertCollectiveInputDeps("RCCL broadcast"))
            return false;

        // Start RCCL group
        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Issue broadcast for each device
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                rccl::ncclGroupEnd();
                return false;
            }

            rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
            hipStream_t stream = static_cast<hipStream_t>(streams_[i]);

            // rcclBroadcast: sendbuff and recvbuff can be the same for in-place
            r = rccl::ncclBroadcast(buffers[i], buffers[i], count,
                                    toRcclDataTypeInt(dtype_int), root, comm, stream);
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rcclBroadcast failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + rccl::ncclGetErrorString(r);
                rccl::ncclGroupEnd();
                return false;
            }
        }

        // End RCCL group
        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion events
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice for event record failed: ") + hipGetErrorString(err);
                return false;
            }

            err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[i]),
                                 static_cast<hipStream_t>(streams_[i]));
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipEventRecord failed: ") + hipGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[RCCLCoordinator] BroadcastMulti completed: " << count << " elements from root " << root);
        collective_performed_.store(true);
        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::doReduceScatterMulti(const std::vector<const void *> &send_buffers,
                                               const std::vector<void *> &recv_buffers,
                                               size_t recv_count, int dtype_int, int op_int)
    {
#ifdef HAVE_RCCL
        if (!doInsertCollectiveInputDeps("RCCL reduce-scatter"))
            return false;

        // Start RCCL group
        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Issue reduce-scatter for each device
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                rccl::ncclGroupEnd();
                return false;
            }

            rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
            hipStream_t stream = static_cast<hipStream_t>(streams_[i]);

            r = rccl::ncclReduceScatter(send_buffers[i], recv_buffers[i], recv_count,
                                        toRcclDataTypeInt(dtype_int), toRcclRedOpInt(op_int),
                                        comm, stream);
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rcclReduceScatter failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + rccl::ncclGetErrorString(r);
                rccl::ncclGroupEnd();
                return false;
            }
        }

        // End RCCL group
        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Record completion events
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = trackedHipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice for event record failed: ") + hipGetErrorString(err);
                return false;
            }

            err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[i]),
                                 static_cast<hipStream_t>(streams_[i]));
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipEventRecord failed: ") + hipGetErrorString(err);
                return false;
            }
        }

        LOG_TRACE("[RCCLCoordinator] ReduceScatterMulti completed: " << recv_count << " elements per device");
        collective_performed_.store(true);
        return true;
#else
        last_error_ = "RCCL not available";
        return false;
#endif
    }

    bool RCCLCoordinator::doCopy(void *dst_ptr, int dst_device_idx,
                                 const void *src_ptr, int src_device_idx,
                                 size_t bytes, bool wait_for_completion,
                                 void *source_stream, void *destination_stream)
    {
#ifdef HAVE_RCCL
        // Start RCCL group for paired send/recv
        rccl::ncclResult_t r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        // Issue send from source device
        hipError_t err = trackedHipSetDevice(device_ordinals_[src_device_idx]);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipSetDevice (src) failed: ") + hipGetErrorString(err);
            rccl::ncclGroupEnd();
            return false;
        }

        rccl::ncclComm_t src_comm = static_cast<rccl::ncclComm_t>(comms_[src_device_idx]);
        hipStream_t src_stream = static_cast<hipStream_t>(source_stream);

        // rcclSend: peer rank is the destination device index within the communicator
        r = rccl::ncclSend(src_ptr, bytes, rccl::ncclInt8, dst_device_idx, src_comm, src_stream);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclSend failed: ") + rccl::ncclGetErrorString(r);
            rccl::ncclGroupEnd();
            return false;
        }

        // Issue recv on destination device
        err = trackedHipSetDevice(device_ordinals_[dst_device_idx]);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipSetDevice (dst) failed: ") + hipGetErrorString(err);
            rccl::ncclGroupEnd();
            return false;
        }

        rccl::ncclComm_t dst_comm = static_cast<rccl::ncclComm_t>(comms_[dst_device_idx]);
        hipStream_t dst_stream = static_cast<hipStream_t>(destination_stream);

        // rcclRecv: peer rank is the source device index within the communicator
        r = rccl::ncclRecv(dst_ptr, bytes, rccl::ncclInt8, src_device_idx, dst_comm, dst_stream);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclRecv failed: ") + rccl::ncclGetErrorString(r);
            rccl::ncclGroupEnd();
            return false;
        }

        // End RCCL group - this enqueues the actual transfer on the streams
        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            return false;
        }

        if (wait_for_completion)
        {
            // Synchronize both streams to ensure copy is complete
            err = trackedHipSetDevice(device_ordinals_[src_device_idx]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice (sync src) failed: ") + hipGetErrorString(err);
                return false;
            }
            err = hipStreamSynchronize(src_stream);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipStreamSynchronize (src) failed: ") + hipGetErrorString(err);
                return false;
            }

            err = trackedHipSetDevice(device_ordinals_[dst_device_idx]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice (sync dst) failed: ") + hipGetErrorString(err);
                return false;
            }
            err = hipStreamSynchronize(dst_stream);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipStreamSynchronize (dst) failed: ") + hipGetErrorString(err);
                return false;
            }
        }

        // Record completion events (for both sync and async paths)
        err = trackedHipSetDevice(device_ordinals_[src_device_idx]);
        if (err == hipSuccess)
        {
            hipError_t evt_err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[src_device_idx]), src_stream);
            if (evt_err != hipSuccess)
            {
                LOG_WARN("[RCCLCoordinator] hipEventRecord (src) failed: " << hipGetErrorString(evt_err));
            }
        }
        err = trackedHipSetDevice(device_ordinals_[dst_device_idx]);
        if (err == hipSuccess)
        {
            hipError_t evt_err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[dst_device_idx]), dst_stream);
            if (evt_err != hipSuccess)
            {
                LOG_WARN("[RCCLCoordinator] hipEventRecord (dst) failed: " << hipGetErrorString(evt_err));
            }
        }

        LOG_DEBUG("[RCCLCoordinator] Copy " << (wait_for_completion ? "completed" : "enqueued")
                                            << ": " << bytes << " bytes from device "
                                            << device_ordinals_[src_device_idx] << " to device " << device_ordinals_[dst_device_idx]);
        collective_performed_.store(true);
        return true;
#else
        (void)dst_ptr;
        (void)dst_device_idx;
        (void)src_ptr;
        (void)src_device_idx;
        (void)bytes;
        (void)wait_for_completion;
        (void)source_stream;
        (void)destination_stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }

} // namespace llaminar2
