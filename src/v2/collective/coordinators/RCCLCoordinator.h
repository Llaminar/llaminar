/**
 * @file RCCLCoordinator.h
 * @brief Dedicated coordinator thread for RCCL collective operations
 * @author David Sanftenberg
 * @date February 2026
 *
 * Owns all rcclComm_t communicators, HIP streams for collectives, and
 * completion events. All RCCL operations (including rcclGroupStart/End)
 * execute on this single thread, ensuring proper threading semantics.
 *
 * This solves the threading mismatch where comms were created on worker
 * threads but used from the caller thread - RCCL requires all operations
 * on a communicator to happen from the same thread.
 *
 * Usage:
 * @code
 *   RCCLCoordinator coord;
 *   coord.initialize({0, 1});  // ROCm GPUs 0 and 1
 *
 *   // From any thread:
 *   coord.allreduceMulti(buffers, count, dtype, op);
 *
 *   // Device worker synchronization:
 *   hipStreamWaitEvent(compute_stream, (hipEvent_t)coord.getCompletionEvent(gpu_idx), 0);
 * @endcode
 *
 * @see ICollectiveCoordinator for the base interface
 * @see docs/v2/sketches/CollectiveCoordinators.md for design documentation
 */

#pragma once

#include "ICollectiveCoordinator.h"
#include "../ICollectiveBackend.h" // For CollectiveDataType, CollectiveOp

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Dedicated coordinator thread for RCCL collective operations
     *
     * Owns all rcclComm_t communicators, HIP streams for collectives, and
     * completion events. All RCCL operations (including rcclGroupStart/End)
     * execute on this single thread, ensuring proper threading semantics.
     *
     * Thread safety: All public methods are thread-safe (work is queued to
     * the coordinator thread for execution).
     */
    class RCCLCoordinator : public ICollectiveCoordinator
    {
    public:
        RCCLCoordinator();
        ~RCCLCoordinator() override;

        // Non-copyable, non-movable (owns thread state)
        RCCLCoordinator(const RCCLCoordinator &) = delete;
        RCCLCoordinator &operator=(const RCCLCoordinator &) = delete;
        RCCLCoordinator(RCCLCoordinator &&) = delete;
        RCCLCoordinator &operator=(RCCLCoordinator &&) = delete;

        // =========================================================================
        // ICollectiveCoordinator interface
        // =========================================================================

        /**
         * @brief Initialize the coordinator with ROCm device ordinals
         *
         * Spawns the coordinator thread, creates per-device HIP streams and events,
         * and initializes RCCL communicators using rcclGroupStart/rcclCommInitRank/rcclGroupEnd.
         *
         * @param device_ordinals ROCm device ordinals (e.g., {0, 1} for GPUs 0 and 1)
         * @return true if initialization succeeded
         */
        bool initialize(const std::vector<int> &device_ordinals) override;

        /**
         * @brief Shutdown the coordinator
         *
         * Signals the coordinator thread to exit, waits for pending work,
         * and releases all RCCL communicators, streams, and events.
         */
        void shutdown() override;

        /**
         * @brief Check if coordinator is initialized and ready
         */
        bool isInitialized() const override { return initialized_.load(); }

        /**
         * @brief Get completion event for a device after last collective
         *
         * Device workers call hipStreamWaitEvent(stream, getCompletionEvent(device), 0)
         * before launching kernels that depend on collective results.
         *
         * @param device_idx Local device index (0 to num_devices-1)
         * @return hipEvent_t cast to void*
         */
        void *getCompletionEvent(int device_idx) const override;

        /**
         * @brief Wait for device worker's stream before collective
         *
         * Coordinator calls this to ensure all prior device work is done
         * before starting a collective that reads from device buffers.
         *
         * @param device_idx Local device index
         * @param worker_event hipEvent_t cast to void*, recorded after device work
         */
        void waitForDeviceEvent(int device_idx, void *worker_event) override;

        // =========================================================================
        // RCCL Collective Operations (thread-safe, queued to coordinator)
        // =========================================================================

        /**
         * @brief In-place allreduce across all local GPUs
         *
         * @param buffers One buffer per device (size = num_devices)
         * @param count Elements per buffer
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool allreduceMulti(const std::vector<void *> &buffers, size_t count,
                            CollectiveDataType dtype, CollectiveOp op);

        /**
         * @brief In-place allreduce across all local GPUs and wait for completion
         *
         * Executes launch + completion wait as one coordinator-queue operation
         * to prevent interleaving between submit calls.
         */
        bool allreduceMultiAndSynchronize(const std::vector<void *> &buffers, size_t count,
                                          CollectiveDataType dtype, CollectiveOp op);

        /**
         * @brief Allgather across all local GPUs
         *
         * Each device contributes send_count elements, receives
         * send_count * num_devices elements in recv buffer.
         *
         * @param send_buffers One send buffer per device
         * @param recv_buffers One recv buffer per device (must hold num_devices * send_count)
         * @param send_count Elements to send from each device
         * @param dtype Data type
         * @return true on success
         */
        bool allgatherMulti(const std::vector<const void *> &send_buffers,
                            const std::vector<void *> &recv_buffers,
                            size_t send_count, CollectiveDataType dtype);

        /**
         * @brief Broadcast from root to all local GPUs
         *
         * @param buffers One buffer per device (root's buffer is source)
         * @param count Elements to broadcast
         * @param dtype Data type
         * @param root Root device index (0 to num_devices-1)
         * @return true on success
         */
        bool broadcastMulti(const std::vector<void *> &buffers, size_t count,
                            CollectiveDataType dtype, int root);

        /**
         * @brief Reduce-scatter across all local GPUs
         *
         * Reduces and scatters, each device receives recv_count elements.
         *
         * @param send_buffers One send buffer per device (must hold recv_count * num_devices)
         * @param recv_buffers One recv buffer per device
         * @param recv_count Elements each device receives
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool reduceScatterMulti(const std::vector<const void *> &send_buffers,
                                const std::vector<void *> &recv_buffers,
                                size_t recv_count, CollectiveDataType dtype,
                                CollectiveOp op);

        /**
         * @brief Single-device allreduce (uses comm_[device_idx])
         *
         * For single-GPU cases or when a specific device needs its own allreduce.
         *
         * @param buffer In-place buffer
         * @param count Elements
         * @param dtype Data type
         * @param op Reduction operation
         * @param device_idx Device index
         * @return true on success
         */
        bool allreduce(void *buffer, size_t count, CollectiveDataType dtype,
                       CollectiveOp op, int device_idx);

        /**
         * @brief Synchronize all RCCL streams (blocking)
         *
         * Waits for all pending RCCL operations to complete on all devices.
         *
         * @return true on success
         */
        bool synchronize();

        /**
         * @brief Copy data between two devices using RCCL send/recv (synchronous)
         *
         * Uses rcclSend/rcclRecv for efficient device-to-device transfers.
         * RCCL internally routes through the best available transport
         * (P2P, NVLink, or host staging if neither is available).
         * Waits for the transfer to complete before returning.
         *
         * @param dst_ptr Destination buffer pointer (device memory)
         * @param dst_device_idx Destination device index (0 to num_devices-1)
         * @param src_ptr Source buffer pointer (device memory)
         * @param src_device_idx Source device index (0 to num_devices-1)
         * @param bytes Number of bytes to copy
         * @return true on success
         */
        bool copy(void *dst_ptr, int dst_device_idx,
                  const void *src_ptr, int src_device_idx,
                  size_t bytes);

        /**
         * @brief Copy data between two devices using RCCL send/recv (asynchronous)
         *
         * Uses rcclSend/rcclRecv for efficient device-to-device transfers.
         * Returns immediately after enqueuing; completion events are recorded
         * on both source and destination devices.
         *
         * Caller should use getCompletionEvent(dst_device_idx) to synchronize:
         * @code
         *   coord.copyAsync(dst, dst_idx, src, src_idx, bytes);
         *   hipStreamWaitEvent(my_stream, (hipEvent_t)coord.getCompletionEvent(dst_idx), 0);
         * @endcode
         *
         * @param dst_ptr Destination buffer pointer (device memory)
         * @param dst_device_idx Destination device index (0 to num_devices-1)
         * @param src_ptr Source buffer pointer (device memory)
         * @param src_device_idx Source device index (0 to num_devices-1)
         * @param bytes Number of bytes to copy
         * @return true on success (transfer enqueued)
         */
        bool copyAsync(void *dst_ptr, int dst_device_idx,
                       const void *src_ptr, int src_device_idx,
                       size_t bytes);

        /**
         * @brief Get last error message
         */
        const std::string &lastError() const { return last_error_; }

        /**
         * @brief Get number of devices managed by this coordinator
         */
        int numDevices() const { return num_devices_; }

        /**
         * @brief Get device ordinal for a local device index
         * @param device_idx Local device index (0 to num_devices-1)
         * @return ROCm device ordinal
         */
        int deviceOrdinal(int device_idx) const
        {
            return device_ordinals_[device_idx];
        }

    protected:
        /**
         * @brief Enqueue work to the coordinator thread
         *
         * Thread-safe: May be called from any thread.
         *
         * @param work Function to execute on coordinator thread
         */
        void enqueueWork(std::function<void()> work) override;

    private:
        // Worker thread main loop
        void coordinatorLoop();

        // Initialization/cleanup on coordinator thread
        void initializeOnThread();
        void cleanupOnThread();

        // Internal collective implementations (called ON coordinator thread)
        bool doAllreduceMulti(const std::vector<void *> &buffers, size_t count,
                              int dtype_int, int op_int);
        bool doSynchronizeAll();
        bool doAllgatherMulti(const std::vector<const void *> &send_buffers,
                              const std::vector<void *> &recv_buffers,
                              size_t send_count, int dtype_int);
        bool doBroadcastMulti(const std::vector<void *> &buffers, size_t count,
                              int dtype_int, int root);
        bool doReduceScatterMulti(const std::vector<const void *> &send_buffers,
                                  const std::vector<void *> &recv_buffers,
                                  size_t recv_count, int dtype_int, int op_int);
        bool doCopy(void *dst_ptr, int dst_device_idx,
                    const void *src_ptr, int src_device_idx,
                    size_t bytes, bool wait_for_completion);

        // State
        std::vector<int> device_ordinals_;
        int num_devices_ = 0;
        std::atomic<bool> initialized_{false};
        std::string last_error_;

        // RCCL state (owned by coordinator thread)
        // Stored as void* to avoid including rccl.h in public header
        std::vector<void *> comms_;             // rcclComm_t[]
        std::vector<void *> streams_;           // hipStream_t[] - one per device
        std::vector<void *> completion_events_; // hipEvent_t[] - signaled after each collective

        // Worker thread
        std::thread coordinator_thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> init_success_{false};
        std::atomic<bool> init_complete_{false};

        // Tracks whether any collective operation was performed.
        // Used to skip ncclCommAbort on unused communicators, which can crash
        // in the ROCm CLR runtime ("Memobj map does not have ptr: 0x0").
        std::atomic<bool> collective_performed_{false};

        // Work queue
        std::queue<std::function<void()>> work_queue_;
        mutable std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
    };

} // namespace llaminar2
