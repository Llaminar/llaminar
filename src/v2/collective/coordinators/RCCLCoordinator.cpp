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
        if (initialized_.load())
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
        LOG_INFO("[RCCLCoordinator] Initialized with " << num_devices_ << " ROCm GPU(s)");
        return true;
#else
        last_error_ = "RCCL not available (HAVE_RCCL not defined)";
        LOG_ERROR("[RCCLCoordinator] " << last_error_);
        return false;
#endif
    }

    void RCCLCoordinator::shutdown()
    {
        if (!initialized_.load() && !running_.load())
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
        LOG_INFO("[RCCLCoordinator] Shutdown complete");
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

        HIP_CHECK_VOID(hipStreamWaitEvent(stream, event, 0));
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

        // Step 2: Generate RCCL unique ID
        rccl::ncclUniqueId unique_id;
        rccl::ncclResult_t r = rccl::ncclGetUniqueId(&unique_id);
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGetUniqueId failed: ") + rccl::ncclGetErrorString(r);
            LOG_ERROR("[RCCLCoordinator] " << last_error_);
            cleanupOnThread();
            return;
        }

        // Step 3: Initialize RCCL communicators using rcclGroupStart/End
        // This is the proper way to initialize multiple communicators from a single thread
        r = rccl::ncclGroupStart();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupStart failed: ") + rccl::ncclGetErrorString(r);
            LOG_ERROR("[RCCLCoordinator] " << last_error_);
            cleanupOnThread();
            return;
        }

        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t hip_err = hipSetDevice(device_ordinals_[i]);
            if (hip_err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice in rcclCommInitRank failed: ") +
                              hipGetErrorString(hip_err);
                LOG_ERROR("[RCCLCoordinator] " << last_error_);
                rccl::ncclGroupEnd();
                cleanupOnThread();
                return;
            }

            rccl::ncclComm_t comm;
            r = rccl::ncclCommInitRank(&comm, num_devices_, unique_id, i);
            if (r != rccl::ncclSuccess)
            {
                last_error_ = std::string("rcclCommInitRank failed for rank ") +
                              std::to_string(i) + ": " + rccl::ncclGetErrorString(r);
                LOG_ERROR("[RCCLCoordinator] " << last_error_);
                rccl::ncclGroupEnd();
                cleanupOnThread();
                return;
            }
            comms_[i] = static_cast<void *>(comm);
            LOG_TRACE("[RCCLCoordinator] Initialized RCCL comm for device " << device_ordinals_[i]);
        }

        r = rccl::ncclGroupEnd();
        if (r != rccl::ncclSuccess)
        {
            last_error_ = std::string("rcclGroupEnd failed: ") + rccl::ncclGetErrorString(r);
            LOG_ERROR("[RCCLCoordinator] " << last_error_);
            cleanupOnThread();
            return;
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
                    hipSetDevice(device_ordinals_[i]);
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
                hipSetDevice(device_ordinals_[i]);
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
                        hipSetDevice(device_ordinals_[i]);
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
                            hipSetDevice(device_ordinals_[i]);
                            hipStreamSynchronize(static_cast<hipStream_t>(streams_[i]));
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
                    hipSetDevice(device_ordinals_[i]);
                    hipFree(prime_bufs[i]);
                }
            }
        }

        // Step 3: Destroy RCCL communicators.
        // After priming (Step 2), all communicators have initialized internal
        // buffers. We use ncclCommDestroy (graceful) rather than ncclCommAbort
        // (aggressive) because ncclCommAbort tears down internal RCCL/ROCm CLR
        // tracking structures in a way that leaves the HIP runtime in an
        // inconsistent state, causing "Memobj map does not have ptr" errors
        // on subsequent communicator creation.
        //
        // NOTE: Even ncclCommDestroy causes slight ROCm CLR state accumulation
        // over many repeated init/destroy cycles (a known ROCm CLR bug). In
        // production inference this is not an issue (single lifecycle). In tests,
        // this can cause crashes after ~14+ cycles. See test fixture caching for
        // the test-side mitigation.
        if (collective_performed_.load())
        {
            for (int i = 0; i < static_cast<int>(comms_.size()); ++i)
            {
                if (comms_[i] != nullptr)
                {
                    if (i < static_cast<int>(device_ordinals_.size()))
                    {
                        hipSetDevice(device_ordinals_[i]);
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
                    hipSetDevice(device_ordinals_[i]);
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
                    hipSetDevice(device_ordinals_[i]);
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

            // Use hipDeviceSynchronize() instead of hipStreamSynchronize(streams_[i])
            // to ensure ALL streams complete, including any internal RCCL streams.
            err = hipDeviceSynchronize();
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipDeviceSynchronize failed for device ") +
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

        // Same device - use hipMemcpy on coordinator thread (synchronous)
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

            err = hipMemcpy(dst_ptr, src_ptr, bytes, hipMemcpyDeviceToDevice);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipMemcpy failed: ") + hipGetErrorString(err);
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
        // CRITICAL: Synchronize ALL streams on each device before RCCL operations.
        // Compute kernels run on a dedicated per-device stream (AMDDeviceContext::default_stream_)
        // created via hipStreamCreateWithFlags — NOT the HIP NULL stream. Using
        // hipStreamSynchronize(nullptr) only syncs the NULL stream, which is a NO-OP
        // for compute work. hipDeviceSynchronize() syncs ALL streams including the
        // dedicated compute stream, ensuring RCCL reads completed data.
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = hipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed during pre-allreduce sync: ") + hipGetErrorString(err);
                return false;
            }
            // Sync ALL streams (compute kernels use a dedicated stream, not NULL stream)
            err = hipDeviceSynchronize();
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipDeviceSynchronize failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                return false;
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

            rccl::ncclComm_t comm = static_cast<rccl::ncclComm_t>(comms_[i]);
            hipStream_t stream = static_cast<hipStream_t>(streams_[i]);

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
        // Sync ALL device streams before RCCL reads buffers (see doAllreduceMulti comment)
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = hipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed during pre-allgather sync: ") + hipGetErrorString(err);
                return false;
            }
            err = hipDeviceSynchronize();
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipDeviceSynchronize failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                return false;
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
        // Sync ALL device streams before RCCL reads buffers (see doAllreduceMulti comment)
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = hipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed during pre-broadcast sync: ") + hipGetErrorString(err);
                return false;
            }
            err = hipDeviceSynchronize();
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipDeviceSynchronize failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                return false;
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
        // Sync ALL device streams before RCCL reads buffers (see doAllreduceMulti comment)
        for (int i = 0; i < num_devices_; ++i)
        {
            hipError_t err = hipSetDevice(device_ordinals_[i]);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed during pre-reducescatter sync: ") + hipGetErrorString(err);
                return false;
            }
            err = hipDeviceSynchronize();
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipDeviceSynchronize failed for device ") +
                              std::to_string(device_ordinals_[i]) + ": " + hipGetErrorString(err);
                return false;
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
            hipEventRecord(static_cast<hipEvent_t>(completion_events_[src_device_idx]), src_stream);
        }
        err = hipSetDevice(device_ordinals_[dst_device_idx]);
        if (err == hipSuccess)
        {
            hipEventRecord(static_cast<hipEvent_t>(completion_events_[dst_device_idx]), dst_stream);
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
