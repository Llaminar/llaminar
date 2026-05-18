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
 * All RCCL operations are serialized on the coordinator thread to ensure
 * proper threading semantics - RCCL requires that all operations on a
 * communicator happen from the same thread.
 */

#include "RCCLCoordinator.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"

#include <functional>

#ifdef HAVE_RCCL
#include <hip/hip_runtime.h>
#include "../backends/RCCLDynamicLoader.h"
// Use the dynamically loaded RCCL types and functions
namespace rccl = llaminar2::rccl_dynamic;
#endif

namespace llaminar2
{

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
        LOG_WARN("[RCCLCoordinator] Aborting all " << comms_.size()
                                                   << " communicators to unblock pending RCCL operations");

        for (int i = 0; i < static_cast<int>(comms_.size()); ++i)
        {
            if (comms_[i] != nullptr)
            {
                if (i < static_cast<int>(device_ordinals_.size()))
                {
                    HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                }
                rccl::ncclResult_t r = rccl::ncclCommAbort(
                    static_cast<rccl::ncclComm_t>(comms_[i]));
                if (r != rccl::ncclSuccess)
                {
                    LOG_WARN("[RCCLCoordinator] ncclCommAbort on device "
                             << i << " returned: " << rccl::ncclGetErrorString(r));
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

        LOG_WARN("[RCCLCoordinator] All communicators aborted");
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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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

    void RCCLCoordinator::cleanupOnThread()
    {
#ifdef HAVE_RCCL
        LOG_TRACE("[RCCLCoordinator] Cleaning up RCCL resources on coordinator thread");

        // Step 1: Synchronize all streams before destroying any resources.
        // This ensures any internally-queued RCCL/HIP work completes before we
        // free communicators, events, or streams.
        for (int i = 0; i < static_cast<int>(streams_.size()); ++i)
        {
            if (streams_[i] != nullptr)
            {
                if (i < static_cast<int>(device_ordinals_.size()))
                {
                    HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                }
                HIP_CHECK_VOID(hipStreamSynchronize(static_cast<hipStream_t>(streams_[i])));
            }
        }

        // Step 2: Prime communicators if no collective was ever performed.
        // RCCL lazily allocates internal work buffers on first collective use.
        // Both ncclCommDestroy and ncclCommAbort on unused communicators trigger
        // crashes in the ROCm CLR ("Memobj map does not have ptr: 0x0") because
        // they try to unmap memory that was never mapped. Simply skipping cleanup
        // leaks RCCL internal state, causing subsequent ncclCommInitRank calls to
        // fail with GPU memory access faults.
        //
        // Solution: perform a trivial 1-element allreduce to force RCCL to allocate
        // its internal buffers, then ncclCommDestroy can safely clean them up.
        if (!collective_performed_.load() && !comms_.empty() && comms_[0] != nullptr)
        {
            LOG_DEBUG("[RCCLCoordinator] Priming " << num_devices_
                                                   << " unused communicators with trivial allreduce before cleanup");

            // Allocate tiny device buffers on each GPU
            std::vector<void *> prime_bufs(num_devices_, nullptr);
            bool alloc_ok = true;
            for (int i = 0; i < num_devices_ && alloc_ok; ++i)
            {
                HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                if (hipMalloc(&prime_bufs[i], sizeof(float)) != hipSuccess)
                {
                    LOG_WARN("[RCCLCoordinator] hipMalloc for prime buffer failed on device "
                             << device_ordinals_[i]);
                    alloc_ok = false;
                }
            }

            if (alloc_ok)
            {
                // Perform trivial allreduce to initialize RCCL internal buffers
                rccl::ncclResult_t r = rccl::ncclGroupStart();
                if (r == rccl::ncclSuccess)
                {
                    bool ops_ok = true;
                    for (int i = 0; i < num_devices_ && ops_ok; ++i)
                    {
                        HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                        r = rccl::ncclAllReduce(
                            prime_bufs[i], prime_bufs[i], 1,
                            rccl::ncclFloat, rccl::ncclSum,
                            static_cast<rccl::ncclComm_t>(comms_[i]),
                            static_cast<hipStream_t>(streams_[i]));
                        if (r != rccl::ncclSuccess)
                        {
                            LOG_WARN("[RCCLCoordinator] Prime allreduce failed: "
                                     << rccl::ncclGetErrorString(r));
                            ops_ok = false;
                        }
                    }
                    rccl::ncclGroupEnd();

                    if (ops_ok)
                    {
                        // Synchronize all streams to ensure the trivial op completes
                        for (int i = 0; i < num_devices_; ++i)
                        {
                            HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                            HIP_CHECK_VOID(hipStreamSynchronize(static_cast<hipStream_t>(streams_[i])));
                        }
                        collective_performed_.store(true);
                        LOG_DEBUG("[RCCLCoordinator] Communicators primed successfully");
                    }
                }
            }

            // Free temporary buffers
            for (int i = 0; i < num_devices_; ++i)
            {
                if (prime_bufs[i] != nullptr)
                {
                    HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                    HIP_CHECK_VOID(hipFree(prime_bufs[i]));
                }
            }
        }

        // Step 3: Release RCCL communicators.
        //
        // RCCL/ROCm CLR has bugs with communicator cleanup on MI60 GPUs:
        // - ncclCommDestroy alone: segfaults inside librccl.so with 4 communicators
        //   created via ncclCommInitAll (null deref at internal struct offset)
        // - ncclCommAbort: corrupts glibc heap metadata, causing SIGABRT during
        //   subsequent cleanup
        //
        // Solution: Use the proper NCCL 2.14+ shutdown sequence:
        //   ncclCommFinalize → hipStreamSynchronize → ncclCommDestroy
        // ncclCommFinalize tells RCCL to flush async operations and prepare
        // internal state for clean destruction.
        if (!comms_.empty())
        {
            bool has_finalize = rccl::isCommFinalizeAvailable();

            if (has_finalize)
            {
                // Step 3a: Finalize all communicators first
                for (int i = 0; i < static_cast<int>(comms_.size()); ++i)
                {
                    if (comms_[i] != nullptr)
                    {
                        if (i < static_cast<int>(device_ordinals_.size()))
                        {
                            HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                        }
                        rccl::ncclResult_t r = rccl::ncclCommFinalize(static_cast<rccl::ncclComm_t>(comms_[i]));
                        if (r != rccl::ncclSuccess)
                        {
                            LOG_WARN("[RCCLCoordinator] ncclCommFinalize failed for device "
                                     << (i < static_cast<int>(device_ordinals_.size()) ? device_ordinals_[i] : -1)
                                     << ": " << rccl::ncclGetErrorString(r));
                        }
                    }
                }

                // Step 3b: Synchronize all streams after finalize
                for (int i = 0; i < static_cast<int>(streams_.size()); ++i)
                {
                    if (streams_[i] != nullptr)
                    {
                        if (i < static_cast<int>(device_ordinals_.size()))
                        {
                            HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                        }
                        HIP_CHECK_VOID(hipStreamSynchronize(static_cast<hipStream_t>(streams_[i])));
                    }
                }

                // Step 3c: Now destroy communicators (safe after finalize+sync)
                for (int i = 0; i < static_cast<int>(comms_.size()); ++i)
                {
                    if (comms_[i] != nullptr)
                    {
                        if (i < static_cast<int>(device_ordinals_.size()))
                        {
                            HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                        }
                        rccl::ncclResult_t r = rccl::ncclCommDestroy(static_cast<rccl::ncclComm_t>(comms_[i]));
                        if (r != rccl::ncclSuccess)
                        {
                            LOG_WARN("[RCCLCoordinator] ncclCommDestroy failed for device "
                                     << (i < static_cast<int>(device_ordinals_.size()) ? device_ordinals_[i] : -1)
                                     << ": " << rccl::ncclGetErrorString(r));
                        }
                        comms_[i] = nullptr;
                    }
                }
            }
            else
            {
                // Fallback: ncclCommFinalize not available — skip cleanup to avoid crash
                LOG_DEBUG("[RCCLCoordinator] ncclCommFinalize not available, skipping comm cleanup "
                          "(OS will reclaim resources at exit)");
                for (int i = 0; i < static_cast<int>(comms_.size()); ++i)
                {
                    comms_[i] = nullptr;
                }
            }
        }
        else
        {
            // Priming failed - last resort: null pointers (leaks RCCL state but avoids crash)
            LOG_WARN("[RCCLCoordinator] Cannot properly clean up communicators - "
                     "priming failed, nulling pointers (may leak RCCL state)");
            for (int i = 0; i < static_cast<int>(comms_.size()); ++i)
            {
                comms_[i] = nullptr;
            }
        }
        comms_.clear();

        // Step 3: Destroy completion events
        for (int i = 0; i < static_cast<int>(completion_events_.size()); ++i)
        {
            if (completion_events_[i] != nullptr)
            {
                if (i < static_cast<int>(device_ordinals_.size()))
                {
                    HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                }
                HIP_CHECK_VOID(hipEventDestroy(static_cast<hipEvent_t>(completion_events_[i])));
                completion_events_[i] = nullptr;
            }
        }
        completion_events_.clear();

        // Step 4: Destroy streams (after comms and events are already gone)
        for (int i = 0; i < static_cast<int>(streams_.size()); ++i)
        {
            if (streams_[i] != nullptr)
            {
                if (i < static_cast<int>(device_ordinals_.size()))
                {
                    HIP_CHECK_VOID(hipSetDevice(device_ordinals_[i]));
                }
                HIP_CHECK_VOID(hipStreamDestroy(static_cast<hipStream_t>(streams_[i])));
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

        // If compute streams aren't registered, fall back to synchronous path
        if (compute_streams_.empty() ||
            static_cast<int>(compute_streams_.size()) != num_devices_)
        {
            LOG_DEBUG("[RCCLCoordinator] allreduceMultiWithComputeDeps: no compute streams, "
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
        err = hipSetDevice(ordinal);
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
            hipError_t err = hipSetDevice(ordinal);
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
        hipError_t err = hipSetDevice(device_ordinals_[device_idx]);
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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
            hipError_t err = hipSetDevice(device_ordinals_[src_device_idx]);
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
                             { return doCopy(dst_ptr, dst_device_idx, src_ptr, src_device_idx, bytes, /*wait_for_completion=*/true); });
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

    bool RCCLCoordinator::copyAsync(void *dst_ptr, int dst_device_idx,
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
            last_error_ = "RCCLCoordinator::copyAsync: null buffer pointer";
            return false;
        }

        if (dst_device_idx < 0 || dst_device_idx >= num_devices_ ||
            src_device_idx < 0 || src_device_idx >= num_devices_)
        {
            last_error_ = "RCCLCoordinator::copyAsync: device index out of range (src=" +
                          std::to_string(src_device_idx) + " dst=" + std::to_string(dst_device_idx) +
                          " num_devices=" + std::to_string(num_devices_) + ")";
            return false;
        }

        // Same device - use hipMemcpyAsync on coordinator thread
        if (src_device_idx == dst_device_idx)
        {
            return submitAndWait([&]()
                                 {
            hipError_t err = hipSetDevice(device_ordinals_[src_device_idx]);
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

            // Record completion event
            err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[src_device_idx]), stream);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipEventRecord failed: ") + hipGetErrorString(err);
                return false;
            }
            return true; });
        }

        // Different devices - use RCCL send/recv without synchronization (async)
        return submitAndWait([&]()
                             { return doCopy(dst_ptr, dst_device_idx, src_ptr, src_device_idx, bytes, /*wait_for_completion=*/false); });
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

    // ============================================================================
    // Internal Collective Implementations (called ON coordinator thread)
    // ============================================================================

    bool RCCLCoordinator::doAllreduceMulti(const std::vector<void *> &buffers, size_t count,
                                           int dtype_int, int op_int)
    {
#ifdef HAVE_RCCL
        const bool trace_device_state = debugEnv().validation.validate_gpu_ptrs;
        const size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());

        // Pre-collective sync: ensure compute kernels have finished writing to
        // the buffers before RCCL reads them.
        //
        // Two modes:
        // (a) Stream-level sync (preferred): Record event on compute stream, then
        //     hipStreamWaitEvent(rccl_stream, compute_event) — zero host stall.
        // (b) Device sync (fallback): hipDeviceSynchronize() — stalls host thread
        //     until all GPU work completes. Used when compute streams aren't registered.
        const bool use_stream_sync = !compute_streams_.empty() &&
                                     static_cast<int>(compute_streams_.size()) == num_devices_;

        for (int i = 0; i < num_devices_; ++i)
        {
            if (trace_device_state)
            {
                int before_dev = -1;
                hipError_t get_before = hipGetDevice(&before_dev);
                if (get_before == hipSuccess)
                {
                    LOG_DEBUG("[RCCL_DEVICE_STATE] phase=pre_sync thread=" << thread_hash
                                                                           << " slot=" << i
                                                                           << " current=" << before_dev
                                                                           << " target=" << device_ordinals_[i]);
                }
            }

            hipError_t err = hipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed during pre-allreduce sync: ") + hipGetErrorString(err);
                return false;
            }

            if (trace_device_state)
            {
                int after_dev = -1;
                hipError_t get_after = hipGetDevice(&after_dev);
                if (get_after == hipSuccess && after_dev != device_ordinals_[i])
                {
                    LOG_ERROR("[RCCL_DEVICE_STATE_MISMATCH] phase=post_set_pre_sync thread=" << thread_hash
                                                                                             << " slot=" << i
                                                                                             << " expected=" << device_ordinals_[i]
                                                                                             << " actual=" << after_dev);
                    last_error_ = "RCCLCoordinator device mismatch after hipSetDevice (pre-sync)";
                    return false;
                }
            }

            if (use_stream_sync)
            {
                // Stream-level pre-sync: record event on compute stream, then
                // make RCCL stream wait for it. Zero host stall.
                hipStream_t compute_stream = static_cast<hipStream_t>(compute_streams_[i]);
                hipEvent_t compute_event = static_cast<hipEvent_t>(compute_events_[i]);
                hipStream_t rccl_stream = static_cast<hipStream_t>(streams_[i]);

                err = hipEventRecord(compute_event, compute_stream);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipEventRecord on compute stream failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }

                err = hipStreamWaitEvent(rccl_stream, compute_event, 0);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipStreamWaitEvent failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
            }
            else
            {
                // Fallback: full device sync (stalls host)
                err = hipDeviceSynchronize();
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipDeviceSynchronize failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
            }
        }

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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
        // Pre-collective sync: ensure compute done before RCCL reads.
        // Uses stream-wait-event if compute streams registered, else device sync.
        const bool use_stream_sync = !compute_streams_.empty() &&
                                     static_cast<int>(compute_streams_.size()) == num_devices_;
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = hipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed during pre-allgather sync: ") + hipGetErrorString(err);
                return false;
            }
            if (use_stream_sync)
            {
                hipStream_t compute_stream = static_cast<hipStream_t>(compute_streams_[i]);
                hipEvent_t compute_event = static_cast<hipEvent_t>(compute_events_[i]);
                hipStream_t rccl_stream = static_cast<hipStream_t>(streams_[i]);
                err = hipEventRecord(compute_event, compute_stream);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipEventRecord failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
                err = hipStreamWaitEvent(rccl_stream, compute_event, 0);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipStreamWaitEvent failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
            }
            else
            {
                err = hipDeviceSynchronize();
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipDeviceSynchronize failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
            }
        }

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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
        // Pre-collective sync: ensure compute done before RCCL reads.
        const bool use_stream_sync = !compute_streams_.empty() &&
                                     static_cast<int>(compute_streams_.size()) == num_devices_;
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = hipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed during pre-broadcast sync: ") + hipGetErrorString(err);
                return false;
            }
            if (use_stream_sync)
            {
                hipStream_t compute_stream = static_cast<hipStream_t>(compute_streams_[i]);
                hipEvent_t compute_event = static_cast<hipEvent_t>(compute_events_[i]);
                hipStream_t rccl_stream = static_cast<hipStream_t>(streams_[i]);
                err = hipEventRecord(compute_event, compute_stream);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipEventRecord failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
                err = hipStreamWaitEvent(rccl_stream, compute_event, 0);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipStreamWaitEvent failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
            }
            else
            {
                err = hipDeviceSynchronize();
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipDeviceSynchronize failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
            }
        }

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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
        // Pre-collective sync: ensure compute done before RCCL reads.
        const bool use_stream_sync = !compute_streams_.empty() &&
                                     static_cast<int>(compute_streams_.size()) == num_devices_;
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = hipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed during pre-reducescatter sync: ") + hipGetErrorString(err);
                return false;
            }
            if (use_stream_sync)
            {
                hipStream_t compute_stream = static_cast<hipStream_t>(compute_streams_[i]);
                hipEvent_t compute_event = static_cast<hipEvent_t>(compute_events_[i]);
                hipStream_t rccl_stream = static_cast<hipStream_t>(streams_[i]);
                err = hipEventRecord(compute_event, compute_stream);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipEventRecord failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
                err = hipStreamWaitEvent(rccl_stream, compute_event, 0);
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipStreamWaitEvent failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
            }
            else
            {
                err = hipDeviceSynchronize();
                if (err != hipSuccess)
                {
                    last_error_ = std::string("hipDeviceSynchronize failed for device ") +
                                  std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                    return false;
                }
            }
        }

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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
            hipError_t err = hipSetDevice(device_ordinals_[i]);
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
                                 size_t bytes, bool wait_for_completion)
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
        hipError_t err = hipSetDevice(device_ordinals_[src_device_idx]);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipSetDevice (src) failed: ") + hipGetErrorString(err);
            rccl::ncclGroupEnd();
            return false;
        }

        rccl::ncclComm_t src_comm = static_cast<rccl::ncclComm_t>(comms_[src_device_idx]);
        hipStream_t src_stream = static_cast<hipStream_t>(streams_[src_device_idx]);

        // rcclSend: peer rank is the destination device index within the communicator
        r = rccl::ncclSend(src_ptr, bytes, rccl::ncclInt8, dst_device_idx, src_comm, src_stream);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclSend failed: ") + rccl::ncclGetErrorString(r);
            rccl::ncclGroupEnd();
            return false;
        }

        // Issue recv on destination device
        err = hipSetDevice(device_ordinals_[dst_device_idx]);
        if (err != hipSuccess)
        {
            last_error_ = std::string("hipSetDevice (dst) failed: ") + hipGetErrorString(err);
            rccl::ncclGroupEnd();
            return false;
        }

        rccl::ncclComm_t dst_comm = static_cast<rccl::ncclComm_t>(comms_[dst_device_idx]);
        hipStream_t dst_stream = static_cast<hipStream_t>(streams_[dst_device_idx]);

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
            err = hipSetDevice(device_ordinals_[src_device_idx]);
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

            err = hipSetDevice(device_ordinals_[dst_device_idx]);
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
        err = hipSetDevice(device_ordinals_[src_device_idx]);
        if (err == hipSuccess)
        {
            hipError_t evt_err = hipEventRecord(static_cast<hipEvent_t>(completion_events_[src_device_idx]), src_stream);
            if (evt_err != hipSuccess)
            {
                LOG_WARN("[RCCLCoordinator] hipEventRecord (src) failed: " << hipGetErrorString(evt_err));
            }
        }
        err = hipSetDevice(device_ordinals_[dst_device_idx]);
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
        last_error_ = "RCCL not available";
        return false;
#endif
    }

} // namespace llaminar2
